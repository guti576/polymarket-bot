# polymarket_collector_5m.py

import requests
import pandas as pd
import numpy as np
import time
import json
import os
import logging
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================

DATASET_FILE    = "./polymarket_dataset_5m.csv"
SLEEP_SECONDS   = 2
ORDERBOOK_DEPTH = 3
MARKET_DURATION = 300        # segundos (5 minutos)
STAKE           = 10.0       # USDC por apuesta para calcular win/loss

# Ask fuera de este rango → mercado sin liquidez real (casi resuelto)
# Estas filas se marcan con market_liquid=0 para filtrarlas en el notebook
ASK_MIN = 0.05
ASK_MAX = 0.95

BINANCE_BASE  = "https://api.binance.com"
BINANCE_PRICE = "https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT"

GAMMA_MARKET_SLUG = "https://gamma-api.polymarket.com/markets/slug/{slug}"
BOOK_URL          = "https://clob.polymarket.com/book"
FEE_RATE_URL      = "https://clob.polymarket.com/fee-rate"

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

session = requests.Session()
session.headers.update({"Connection": "keep-alive"})


# =========================
# TOKEN MAPPING
# =========================

def parse_listish(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x) for x in arr]
            except Exception:
                pass
        if "," in s:
            return [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]
        return [s]
    return [str(v)]


def map_tokens_up_down(market: dict) -> tuple:
    """
    Identifica correctamente cuál token_id corresponde a UP y cuál a DOWN
    inspeccionando el campo 'outcomes'. Evita asumir que token_ids[0] = UP.
    """
    outcomes  = parse_listish(market.get("outcomes"))
    token_ids = parse_listish(market.get("clobTokenIds"))

    if len(outcomes) != len(token_ids) or len(token_ids) < 2:
        raise ValueError(f"No pude mapear outcomes={outcomes} token_ids={token_ids}")

    up_id = down_id = None
    up_lab = down_lab = None

    for o, tid in zip(outcomes, token_ids):
        ol   = o.strip().lower()
        tidn = str(tid).strip()
        if ol in ("up", "yes", "true"):
            up_id,   up_lab   = tidn, o
        elif ol in ("down", "no", "false"):
            down_id, down_lab = tidn, o

    if up_id is None and down_id is None:
        up_id,   down_id   = str(token_ids[0]).strip(), str(token_ids[1]).strip()
        up_lab,  down_lab  = outcomes[0], outcomes[1]

    return up_id, down_id, up_lab, down_lab


# =========================
# BTC SPOT
# =========================

def get_btc_price() -> float | None:
    try:
        r = session.get(BINANCE_PRICE, timeout=5).json()
        return float(r["price"])
    except Exception as e:
        log.warning(f"get_btc_price failed: {e}")
        return None


# =========================
# POLYMARKET MARKET INFO
# =========================

def get_current_market():
    now   = int(time.time())
    epoch = now - (now % MARKET_DURATION)

    for candidate_epoch in [epoch, epoch + MARKET_DURATION]:
        slug = f"btc-updown-5m-{candidate_epoch}"
        try:
            market = session.get(
                GAMMA_MARKET_SLUG.format(slug=slug), timeout=5
            ).json()

            up_id, down_id, _, _ = map_tokens_up_down(market)
            seconds_left = max(0, candidate_epoch + MARKET_DURATION - int(time.time()))

            return slug, up_id, down_id, seconds_left, candidate_epoch

        except Exception:
            continue

    log.warning("get_current_market: no active 5m market found")
    return None, None, None, None, None


# =========================
# POLYMARKET ORDERBOOK
# =========================

def get_orderbook_snapshot(token_id: str, prefix: str) -> tuple[dict, list]:
    """
    Captura ORDERBOOK_DEPTH niveles de bid y ask.
    Devuelve (row_dict, ask_levels) donde ask_levels es la lista completa
    de (price, size) ordenada, lista para calc_fill sin coste HTTP extra.

    BUG FIX: ordena bids y asks explícitamente porque la API no garantiza orden.
    """
    row        = {}
    ask_levels = []

    try:
        r    = session.get(f"{BOOK_URL}?token_id={token_id}", timeout=5).json()
        bids = sorted(r.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
        asks = sorted(r.get("asks", []), key=lambda x: float(x["price"]))

        for i in range(ORDERBOOK_DEPTH):
            row[f"{prefix}_bid_p_{i+1}"] = float(bids[i]["price"]) if i < len(bids) else None
            row[f"{prefix}_bid_s_{i+1}"] = float(bids[i]["size"])  if i < len(bids) else None
            row[f"{prefix}_ask_p_{i+1}"] = float(asks[i]["price"]) if i < len(asks) else None
            row[f"{prefix}_ask_s_{i+1}"] = float(asks[i]["size"])  if i < len(asks) else None

        ask_levels = [(float(a["price"]), float(a["size"])) for a in asks]

    except Exception as e:
        log.warning(f"get_orderbook_snapshot ({prefix}) failed: {e}")

    return row, ask_levels


# =========================
# FEES
# =========================

def get_fee_rate(token_id: str) -> int:
    """
    BUG FIX: La API devuelve 'base_fee' = 1000, que corresponde a
    1000 / 100 = 10 bps = 0.10% (no 10%).
    La escala del campo es: valor / 100 = bps reales.

    Polymarket cobra ~0-20 bps según el mercado.
    Se añade un sanity check: si el valor supera 500 bps (5%) es
    casi seguro un error de parsing → loggear y devolver 0.
    """
    try:
        r       = session.get(f"{FEE_RATE_URL}?token_id={token_id}", timeout=5).json()
        raw_val = r.get("fee_rate") or r.get("base_fee") or r.get("feeRate") or 0
        fee_bps = int(float(raw_val)) // 100   # BUG FIX: dividir entre 100

        if fee_bps > 500:
            log.warning(f"fee_bps={fee_bps} parece anormalmente alto (raw={raw_val}). Usando 0.")
            return 0

        return fee_bps

    except Exception as e:
        log.warning(f"get_fee_rate failed: {e}")
        return 0


# =========================
# FILL SIMULATION
# =========================

def calc_fill(ask_levels: list, fee_bps: int, stake: float = STAKE) -> dict:
    """
    Simula una market order de `stake` USDC recorriendo los niveles ask,
    igual que la UI de Polymarket.

    Columnas en el CSV:
      {prefix}_avg_fill   precio medio ponderado de ejecución real
      {prefix}_win_net    ganancia neta si aciertas (USDC), fees incluidas
      {prefix}_loss_net   pérdida neta si fallas (USDC), fees incluidas, siempre negativo
      {prefix}_break_even win rate mínimo para EV = 0
    """
    if not ask_levels:
        return {"avg_fill": None, "win_net": None, "loss_net": None, "break_even": None}

    fee_pct      = fee_bps / 10_000
    fee_usdc     = stake * fee_pct
    remaining    = stake
    total_shares = 0.0

    for price, size in ask_levels:
        if remaining <= 0:
            break
        usdc_avail    = size * price
        usdc_used     = min(remaining, usdc_avail)
        total_shares += usdc_used / price
        remaining    -= usdc_used

    usdc_filled = stake - remaining
    if usdc_filled <= 0 or total_shares <= 0:
        return {"avg_fill": None, "win_net": None, "loss_net": None, "break_even": None}

    avg_fill   = round(usdc_filled / total_shares, 6)
    win_net    = round(total_shares - usdc_filled - fee_usdc, 6)
    loss_net   = round(-usdc_filled - fee_usdc, 6)
    break_even = (
        round(abs(loss_net) / (win_net + abs(loss_net)), 6)
        if win_net > 0 else 1.0
    )

    return {
        "avg_fill":   avg_fill,
        "win_net":    win_net,
        "loss_net":   loss_net,
        "break_even": break_even,
    }


# =========================
# BINANCE DEPTH
# =========================

def get_binance_depth() -> dict:
    try:
        r    = session.get(f"{BINANCE_BASE}/api/v3/depth?symbol=BTCUSDT&limit=20", timeout=5).json()
        bids = np.array([[float(p), float(q)] for p, q in r["bids"]])
        asks = np.array([[float(p), float(q)] for p, q in r["asks"]])

        bid_vol  = bids[:10, 1].sum()
        ask_vol  = asks[:10, 1].sum()
        best_bid = bids[0, 0]
        best_ask = asks[0, 0]

        return {
            "bid_volume_10":       bid_vol,
            "ask_volume_10":       ask_vol,
            "orderbook_imbalance": (bid_vol - ask_vol) / (bid_vol + ask_vol),
            "mid_price":           (best_bid + best_ask) / 2,
            "spread":              best_ask - best_bid,
        }
    except Exception as e:
        log.warning(f"get_binance_depth failed: {e}")
        return {}


# =========================
# KLINES FEATURES
# =========================

def get_klines_features() -> dict:
    """
    Klines de 1 minuto. Usa iloc[-2] (última vela CERRADA).
    Las features cambian una vez por minuto — es comportamiento normal,
    no un bug. La resolución sub-minuto viene del order book y trade flow.
    """
    try:
        r = session.get(
            f"{BINANCE_BASE}/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100",
            timeout=5,
        ).json()

        df = pd.DataFrame(r, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "qa_vol", "n_trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        price   = df["close"].iloc[-2]
        returns = df["close"].pct_change()

        ema3  = df["close"].ewm(span=3).mean().iloc[-2]
        ema12 = df["close"].ewm(span=12).mean().iloc[-2]

        return {
            "ret_1m":         returns.iloc[-2],
            "ret_3m":         df["close"].pct_change(3).iloc[-2],
            "ret_5m":         df["close"].pct_change(5).iloc[-2],
            "ret_10m":        df["close"].pct_change(10).iloc[-2],
            "ema_3":          ema3,
            "ema_6":          df["close"].ewm(span=6).mean().iloc[-2],
            "ema_12":         ema12,
            "ema_24":         df["close"].ewm(span=24).mean().iloc[-2],
            "ema_ratio":      ema3 / ema12,
            "price_vs_ema12": price / ema12,
            "volume_1m":      df["volume"].iloc[-2],
            "volume_3m":      df["volume"].iloc[-4:-1].sum(),
            "volume_5m":      df["volume"].iloc[-6:-1].sum(),
            "volatility_3m":  returns.iloc[-4:-1].std(),
            "volatility_5m":  returns.iloc[-6:-1].std(),
        }

    except Exception as e:
        log.warning(f"get_klines_features failed: {e}")
        return {}


# =========================
# TRADE FLOW
# =========================

def get_trade_flow() -> dict:
    """
    BUG FIX: ret_30s eliminado.

    Motivo: los últimos 200 trades de BTC/USDT ocurren todos dentro de
    los últimos 10-15 segundos — el cutoff a 30s nunca se alcanza con
    limit=200, por lo que ret_30s era siempre NaN. Solucionarlo requeriría
    limit=1000, añadiendo ~500ms de latencia por ciclo.

    La señal de momentum sub-minuto ya está cubierta por:
      · orderbook_imbalance  (presión en el libro en tiempo real)
      · trade_imbalance      (flujo agresivo de los últimos 200 trades)
      · ret_1m               (último minuto cerrado de klines)
    """
    try:
        r    = session.get(f"{BINANCE_BASE}/api/v3/trades?symbol=BTCUSDT&limit=200", timeout=5).json()
        buy  = 0.0
        sell = 0.0

        for t in r:
            qty = float(t["qty"])
            if t["isBuyerMaker"]:
                sell += qty
            else:
                buy  += qty

        total = buy + sell
        return {
            "buy_volume_last200":  buy,
            "sell_volume_last200": sell,
            "trade_imbalance":     (buy - sell) / total if total > 0 else 0,
        }

    except Exception as e:
        log.warning(f"get_trade_flow failed: {e}")
        return {}


# =========================
# MAIN LOOP
# =========================

def main():
    log.info("collector started — 5m markets")
    log.info(
        f"sleep={SLEEP_SECONDS}s | "
        f"~{MARKET_DURATION // (SLEEP_SECONDS + 2)} snapshots/market | "
        f"stake={STAKE} USDC"
    )

    current_slug      = None
    market_open_price = None
    cached_up_fee     = 0
    cached_down_fee   = 0

    while True:
        try:
            slug, tk_up, tk_down, secs_left, epoch = get_current_market()

            if slug is None:
                time.sleep(SLEEP_SECONDS)
                continue

            btc_price = get_btc_price()

            # ── Nuevo mercado ─────────────────────────────────────────────────
            if slug != current_slug:
                log.info(f"new market: {slug}  (epoch={epoch})")
                current_slug      = slug
                market_open_price = btc_price if btc_price is not None else None

                if btc_price is None:
                    log.warning("btc_price unavailable at market open — will retry")

                cached_up_fee   = get_fee_rate(tk_up)
                cached_down_fee = get_fee_rate(tk_down)
                log.info(f"fees — up: {cached_up_fee} bps  down: {cached_down_fee} bps")

            if market_open_price is None and btc_price is not None:
                market_open_price = btc_price
                log.info(f"market_open_price recovered: {market_open_price}")

            # ── Order book ───────────────────────────────────────────────────
            # BUG FIX: si UP falla, se reintenta una vez antes de guardar NaN
            ob_up,   asks_up   = get_orderbook_snapshot(tk_up,   "up")
            if not asks_up:
                log.warning("UP book empty — retrying once")
                time.sleep(0.5)
                ob_up, asks_up = get_orderbook_snapshot(tk_up, "up")

            ob_down, asks_down = get_orderbook_snapshot(tk_down, "down")

            # ── Construir row ─────────────────────────────────────────────────
            elapsed         = MARKET_DURATION - secs_left
            market_progress = round(elapsed / MARKET_DURATION, 4)

            row = {
                "timestamp":             datetime.now(timezone.utc).isoformat(),
                "market_slug":           slug,
                "seconds_left":          secs_left,
                "market_progress":       market_progress,
                "btc_spot":              btc_price,
                "btc_price_market_open": market_open_price,
                "btc_return_since_open": (
                    btc_price / market_open_price - 1
                    if (btc_price and market_open_price) else None
                ),
            }

            row.update(ob_up)
            row.update(ob_down)

            row["up_fee_bps"]   = cached_up_fee
            row["down_fee_bps"] = cached_down_fee

            # ── Fill simulation ───────────────────────────────────────────────
            fill_up   = calc_fill(asks_up,   cached_up_fee,   STAKE)
            fill_down = calc_fill(asks_down, cached_down_fee, STAKE)

            row["up_avg_fill"]   = fill_up["avg_fill"]
            row["up_win_net"]    = fill_up["win_net"]
            row["up_loss_net"]   = fill_up["loss_net"]
            row["up_break_even"] = fill_up["break_even"]

            row["down_avg_fill"]   = fill_down["avg_fill"]
            row["down_win_net"]    = fill_down["win_net"]
            row["down_loss_net"]   = fill_down["loss_net"]
            row["down_break_even"] = fill_down["break_even"]

            # BUG FIX: flag de liquidez real
            # market_liquid=1 → asks en rango 0.05-0.95, mercado abierto y tradeable
            # market_liquid=0 → mercado casi resuelto, no usar para entrenar ni tradear
            ask_up_1  = row.get("up_ask_p_1")
            ask_dn_1  = row.get("down_ask_p_1")
            row["market_liquid"] = int(
                ask_up_1 is not None and ASK_MIN <= ask_up_1 <= ASK_MAX and
                ask_dn_1 is not None and ASK_MIN <= ask_dn_1 <= ASK_MAX
            )

            # ── Binance ───────────────────────────────────────────────────────
            row.update(get_binance_depth())
            row.update(get_klines_features())
            row.update(get_trade_flow())

            # ── Guardar CSV ───────────────────────────────────────────────────
            file_exists = os.path.isfile(DATASET_FILE)
            pd.DataFrame([row]).to_csv(
                DATASET_FILE,
                mode   = "a",
                header = not file_exists,
                index  = False,
            )

            log.info(
                f"saved | secs={secs_left:>3}s"
                f" | prog={market_progress:.0%}"
                f" | liquid={row['market_liquid']}"
                f" | up_win={fill_up['win_net']}"
                f" | dn_win={fill_down['win_net']}"
                f" | btc={btc_price}"
            )

        except Exception as e:
            log.error(f"main loop error: {e}", exc_info=True)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()