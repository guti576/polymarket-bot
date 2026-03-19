# polymarket_live_opportunity.py

import json
import os
import time
import requests
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================

STAKE           = 10.0   # USDC por apuesta
REFRESH_SECONDS = 4
MARKET_DURATION = 300    # segundos (5 minutos)

# Ask por encima de este umbral → libro sin liquidez real
ASK_LIQUIDITY_THRESHOLD = 0.80

GAMMA_MARKET_BY_SLUG = "https://gamma-api.polymarket.com/markets/slug/{slug}"
GAMMA_EVENTS_SLUG    = "https://gamma-api.polymarket.com/events/slug/{slug}"
CLOB_BOOK            = "https://clob.polymarket.com/book"
BINANCE_PRICE        = "https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT"

session = requests.Session()
session.headers.update({"Connection": "keep-alive"})


# =========================
# FETCH LOGIC  (del código que funciona)
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


def map_tokens_up_down(market):
    outcomes  = parse_listish(market.get("outcomes"))
    token_ids = parse_listish(market.get("clobTokenIds"))

    if len(outcomes) != len(token_ids) or len(token_ids) < 2:
        raise ValueError("No pude mapear outcomes y token_ids")

    up_id = down_id = None
    up_lab = down_lab = None

    for o, tid in zip(outcomes, token_ids):
        ol   = o.strip().lower()
        tidn = str(tid).strip()
        if ol in ("up", "yes", "true"):
            up_id, up_lab = tidn, o
        elif ol in ("down", "no", "false"):
            down_id, down_lab = tidn, o

    if up_id is None and down_id is None:
        up_id,   down_id   = str(token_ids[0]).strip(), str(token_ids[1]).strip()
        up_lab,  down_lab  = outcomes[0], outcomes[1]

    return up_id, down_id, up_lab, down_lab


def fetch_book(token_id):
    """
    Devuelve todos los niveles bid y ask del libro, ordenados correctamente.
    """
    r = session.get(
        CLOB_BOOK,
        params={"token_id": token_id, "_t": int(time.time())},
        timeout=5,
    )
    r.raise_for_status()
    data = r.json()

    bids = sorted(
        [(float(x["price"]), float(x["size"])) for x in data.get("bids") or []],
        key=lambda x: x[0], reverse=True
    )
    asks = sorted(
        [(float(x["price"]), float(x["size"])) for x in data.get("asks") or []],
        key=lambda x: x[0]
    )
    return bids, asks


def get_current_slug() -> str | None:
    """
    Construye el slug del mercado de 5m activo y verifica que existe.
    Prueba el epoch actual y el siguiente por si el mercado nuevo
    ya está publicado antes de que cambie el reloj local.
    """
    now   = int(time.time())
    epoch = now - (now % MARKET_DURATION)

    for candidate in [epoch, epoch + MARKET_DURATION]:
        slug = f"btc-updown-5m-{candidate}"
        try:
            r = session.get(
                GAMMA_EVENTS_SLUG.format(slug=slug), timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("markets"):
                    return slug
        except Exception:
            continue
    return None


def get_market_data(slug: str) -> dict:
    """
    Obtiene los datos completos del mercado: token ids, order book,
    métricas de oportunidad y probabilidad implícita.
    """
    market = session.get(
        GAMMA_MARKET_BY_SLUG.format(slug=slug), timeout=5
    ).json()

    up_id, down_id, up_lab, down_lab = map_tokens_up_down(market)

    up_bids,   up_asks   = fetch_book(up_id)
    down_bids, down_asks = fetch_book(down_id)

    def side_info(bids, asks, label, token_id):
        best_bid_p, best_bid_s = bids[0] if bids else (None, None)
        best_ask_p, best_ask_s = asks[0] if asks else (None, None)

        mid = (
            (best_bid_p + best_ask_p) / 2.0
            if (best_bid_p is not None and best_ask_p is not None) else None
        )

        # Simular market order de STAKE USDC recorriendo niveles ask
        remaining    = STAKE
        total_shares = 0.0
        levels_used  = []

        for price, size in asks:
            if remaining <= 0:
                break
            usdc_avail   = size * price
            usdc_used    = min(remaining, usdc_avail)
            shares       = usdc_used / price
            total_shares += shares
            remaining    -= usdc_used
            levels_used.append((price, round(usdc_used, 4)))

        usdc_filled = STAKE - remaining
        fill_pct    = usdc_filled / STAKE
        avg_price   = usdc_filled / total_shares if total_shares > 0 else None
        win_net     = round(total_shares - usdc_filled, 4) if total_shares > 0 else None
        loss_net    = round(-usdc_filled, 4)
        break_even  = (
            abs(loss_net) / (win_net + abs(loss_net))
            if (win_net is not None and win_net > 0) else 1.0
        )
        ev_50 = round(0.5 * win_net + 0.5 * loss_net, 4) if win_net is not None else None

        return {
            "label":        label,
            "token_id":     token_id,
            "best_bid":     best_bid_p,
            "best_bid_s":   best_bid_s,
            "best_ask":     best_ask_p,
            "best_ask_s":   best_ask_s,
            "mid":          mid,
            "avg_price":    round(avg_price, 4) if avg_price else None,
            "total_shares": round(total_shares, 4),
            "fill_pct":     fill_pct,
            "win_net":      win_net,
            "loss_net":     loss_net,
            "break_even":   break_even,
            "ev_50":        ev_50,
            "levels_used":  levels_used,
        }

    up   = side_info(up_bids,   up_asks,   up_lab,   up_id)
    down = side_info(down_bids, down_asks, down_lab, down_id)

    # Probabilidad implícita de UP (media del mid UP y el inverso del mid DOWN)
    p_up = None
    if up["mid"] is not None and down["mid"] is not None:
        p_up = (up["mid"] + (1.0 - down["mid"])) / 2.0
    elif up["mid"] is not None:
        p_up = up["mid"]
    elif down["mid"] is not None:
        p_up = 1.0 - down["mid"]

    return {
        "slug":     slug,
        "question": market.get("question", slug),
        "up":       up,
        "down":     down,
        "p_up":     p_up,
    }


def get_btc_price() -> float | None:
    try:
        return float(session.get(BINANCE_PRICE, timeout=5).json()["price"])
    except Exception:
        return None


def has_liquidity(side: dict) -> bool:
    return (
        side["best_ask"] is not None and
        side["best_ask"] < ASK_LIQUIDITY_THRESHOLD
    )


# =========================
# RENDER
# =========================

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def render_waiting(slug, secs_left, btc, reason):
    progress_pct = (MARKET_DURATION - secs_left) / MARKET_DURATION
    bar          = "█" * int(30 * progress_pct) + "░" * int(30 * (1 - progress_pct))
    now_str      = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    clear()
    print("═" * 62)
    print("  POLYMARKET — BTC UP/DOWN 5m   LIVE OPPORTUNITY")
    print("═" * 62)
    print(f"  Mercado  : {slug or '...'}")
    print(f"  BTC spot : {btc or 'N/A'} USDT")
    print(f"  Tiempo   : [{bar}] {secs_left:>3}s restantes" if secs_left else f"  Tiempo   : calculando...")
    print(f"  {now_str}")
    print()
    print(f"  ⏳ {reason}")
    print()
    print(f"  Actualizando cada {REFRESH_SECONDS}s  |  Ctrl+C para salir")
    print("═" * 62)


def render(data: dict, secs_left: int, btc: float | None):
    slug         = data["slug"]
    up           = data["up"]
    dn           = data["down"]
    p_up         = data["p_up"]
    progress_pct = (MARKET_DURATION - secs_left) / MARKET_DURATION
    bar          = "█" * int(30 * progress_pct) + "░" * int(30 * (1 - progress_pct))
    now_str      = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    def f(val, suffix="", dec=4):
        return f"{val:.{dec}f}{suffix}" if val is not None else "N/A"

    clear()
    print("═" * 62)
    print("  POLYMARKET — BTC UP/DOWN 5m   LIVE OPPORTUNITY")
    print("═" * 62)
    print(f"  {data['question']}")
    print(f"  BTC spot : {btc or 'N/A'} USDT")
    print(f"  Tiempo   : [{bar}] {secs_left:>3}s restantes")
    print(f"  Progreso : {progress_pct:.0%}  |  {now_str}")
    print(f"  Stake    : {STAKE} USDC")
    if p_up is not None:
        print(f"  P(UP) implícita del mercado: {p_up:.1%}")
    print()

    rows = [
        ("",                        f"{'UP':>14}",              f"{'DOWN':>14}"),
        ("─" * 26,                  "─" * 14,                   "─" * 14),
        ("Best bid",                f(up["best_bid"]),           f(dn["best_bid"])),
        ("Best ask (entrada)",      f(up["best_ask"]),           f(dn["best_ask"])),
        ("Liquidez en best ask",    f(up["best_ask_s"], " USDC", 2), f(dn["best_ask_s"], " USDC", 2)),
        ("Avg fill price ★",        f(up["avg_price"]),          f(dn["avg_price"])),
        ("Shares compradas",        f(up["total_shares"], " sh", 2), f(dn["total_shares"], " sh", 2)),
        ("Liquidez cubierta",       f"{up['fill_pct']:.0%}",     f"{dn['fill_pct']:.0%}"),
        ("─" * 26,                  "─" * 14,                   "─" * 14),
        ("Ganas si aciertas ✅",    f"{up['win_net']:+.2f} USDC" if up["win_net"] else "N/A",
                                    f"{dn['win_net']:+.2f} USDC" if dn["win_net"] else "N/A"),
        ("Pierdes si fallas ❌",    f"{up['loss_net']:+.2f} USDC",  f"{dn['loss_net']:+.2f} USDC"),
        ("─" * 26,                  "─" * 14,                   "─" * 14),
        ("Break-even win rate",     f"{up['break_even']:.1%}",   f"{dn['break_even']:.1%}"),
        ("EV (sin modelo, p=50%)",  f"{up['ev_50']:+.4f}" if up["ev_50"] else "N/A",
                                    f"{dn['ev_50']:+.4f}" if dn["ev_50"] else "N/A"),
    ]

    for label, val_up, val_dn in rows:
        if label.startswith("─"):
            print("  " + "─" * 56)
        elif label == "":
            print(f"  {'':26} {val_up:>14} {val_dn:>14}")
        else:
            print(f"  {label:<26} {val_up:>14} {val_dn:>14}")

    print()

    # Detalle del fill si se usaron varios niveles del libro
    for side_label, side in [("UP", up), ("DOWN", dn)]:
        if len(side["levels_used"]) > 1:
            parts = [f"{u}$ @ {p}" for p, u in side["levels_used"]]
            print(f"  Fill {side_label}: " + "  +  ".join(parts))

    print()

    # Avisos por lado
    for side_label, side in [("UP", up), ("DOWN", dn)]:
        if side["win_net"] is None or side["win_net"] <= 0:
            print(f"  ⛔ {side_label}: imposible EV positivo (payout ≤ stake)")
        elif side["fill_pct"] < 1.0:
            print(f"  ⚠️  {side_label}: liquidez insuficiente — solo {side['fill_pct']:.0%} del stake cubierto")
        elif side["break_even"] > 0.70:
            print(f"  ⚠️  {side_label}: break-even {side['break_even']:.1%} — necesitas mucho edge")
        elif side["break_even"] < 0.58:
            print(f"  ✅ {side_label}: break-even {side['break_even']:.1%} — rango razonable")

    print()
    print(f"  ★ avg fill = precio medio real ejecutando {STAKE} USDC market order")
    print(f"  Actualizando cada {REFRESH_SECONDS}s  |  Ctrl+C para salir")
    print("═" * 62)


# =========================
# MAIN LOOP
# =========================

def main():
    print("Conectando con Polymarket...")

    current_slug = None

    while True:
        try:
            # Detectar mercado activo
            slug = get_current_slug()
            btc  = get_btc_price()

            if slug is None:
                render_waiting(None, 0, btc, "Sin mercado activo. Reintentando...")
                time.sleep(REFRESH_SECONDS)
                continue

            # Calcular seconds_left
            epoch     = int(slug.split("-")[-1])
            secs_left = max(0, epoch + MARKET_DURATION - int(time.time()))

            if slug != current_slug:
                current_slug = slug

            # Obtener datos del mercado
            data = get_market_data(slug)
            up   = data["up"]
            dn   = data["down"]

            # Sin liquidez real → pantalla de espera
            if not has_liquidity(up) and not has_liquidity(dn):
                best = up["best_ask"] or "?"
                render_waiting(
                    slug, secs_left, btc,
                    f"Esperando liquidez — best ask UP: {best} "
                    f"(umbral < {ASK_LIQUIDITY_THRESHOLD}). "
                    f"Los market makers suelen entrar 20-60s tras la apertura."
                )
                time.sleep(REFRESH_SECONDS)
                continue

            render(data, secs_left, btc)

        except KeyboardInterrupt:
            print("\nSaliendo.")
            break
        except Exception as e:
            clear()
            print(f"Error: {e}")

        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()