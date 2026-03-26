#!/usr/bin/env python3
"""
order_executor.py — Ejecutor de órdenes Polymarket BTC

Lee trades_5m.csv incrementalmente y ejecuta las señales:
    LONG_UP  / LONG_DOWN  → comprar shares
    EXIT_UP  / EXIT_DOWN  → vender shares

Usa la API CLOB de Polymarket (py-clob-client).
Resuelve market_slug → token_id vía API.

Uso:
    python order_executor.py                    # dry-run (por defecto)
    python order_executor.py --live             # ejecución real
    python order_executor.py --live --stake 5   # stake custom

Requisitos:
    pip install py-clob-client pandas requests
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("executor")


# =============================================================================
# Configuración — EDITAR AQUÍ
# =============================================================================
CFG = dict(
    # ── Polymarket API CLOB ──────────────────────────────────────────────
    API_KEY        = "5e35a342-02cb-a516-c07a-fe19e7421a90",
    API_SECRET     = "GfahqsU1orpvNnxb2DRly_1YaQM9adrkODwHSVysJfI=",
    API_PASSPHRASE = "07277fdb71680b13fae5f1e88b728616da11d1cbef350621b44dcb1a24407f21",
    PRIVATE_KEY    = "7f4eab4d4ff6348a8c0c60327b7c1d9ac9e5535a795164cf8122444bee9fe13a",        # sin 0x

    # ── Red ──────────────────────────────────────────────────────────────
    CLOB_HOST      = "https://clob.polymarket.com",
    CHAIN_ID       = 137,                          # Polygon mainnet

    # ── Trading ──────────────────────────────────────────────────────────
    STAKE          = 2.0,      # USDC por trade
    SLIPPAGE       = 0.03,      # 3% slippage máximo para market orders

    # ── Paths ────────────────────────────────────────────────────────────
    SIGNALS_PATH   = "trades_5m.csv",
    EXEC_LOG_PATH  = "executed_orders.csv",
)


# =============================================================================
# Cache de market_slug → token_ids
# =============================================================================
class MarketResolver:
    """
    Resuelve market_slug → {up_token_id, down_token_id} vía gamma-api.polymarket.com.
    Cachea resultados para no repetir llamadas.
    """

    GAMMA_API = "https://gamma-api.polymarket.com"

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._cache_file = "market_cache.json"
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file) as f:
                    self._cache = json.load(f)
                log.info("Cache de mercados: %d entradas", len(self._cache))
            except Exception:
                pass

    def _save_cache(self):
        try:
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass

    def resolve(self, market_slug: str) -> dict | None:
        """
        Retorna {'up_token_id': str, 'down_token_id': str, 'condition_id': str}
        o None si no se puede resolver.

        Usa gamma-api.polymarket.com que devuelve:
            outcomes:     '["Up", "Down"]'
            clobTokenIds: '["token_up", "token_down"]'
        """
        if market_slug in self._cache:
            return self._cache[market_slug]

        try:
            url = f"{self.GAMMA_API}/markets"
            resp = requests.get(url, params={"slug": market_slug}, timeout=10)

            if resp.status_code != 200:
                log.warning("API error resolviendo %s: HTTP %d", market_slug, resp.status_code)
                return None

            data = resp.json()

            # La API devuelve una lista — coger el primer resultado
            if isinstance(data, list):
                if len(data) == 0:
                    log.warning("Mercado %s no encontrado en API", market_slug)
                    return None
                data = data[0]

            condition_id = data.get("conditionId", "")

            # outcomes y clobTokenIds son strings JSON → parsear
            outcomes_raw = data.get("outcomes", "[]")
            tokens_raw = data.get("clobTokenIds", "[]")

            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
            token_ids = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw

            if len(outcomes) != len(token_ids) or len(outcomes) < 2:
                log.warning("Mercado %s: outcomes/tokens no válidos: %s / %s",
                            market_slug, outcomes, token_ids)
                return None

            # Mapear outcome → token_id
            result = {"condition_id": condition_id}
            for outcome, token_id in zip(outcomes, token_ids):
                key = outcome.strip().lower()
                if key in ("up", "yes"):
                    result["up_token_id"] = token_id
                elif key in ("down", "no"):
                    result["down_token_id"] = token_id

            if "up_token_id" not in result or "down_token_id" not in result:
                log.warning("No se encontraron tokens UP/DOWN para %s (outcomes=%s)",
                            market_slug, outcomes)
                return None

            self._cache[market_slug] = result
            self._save_cache()
            log.info("📍 Resuelto %s → UP=%s... DOWN=%s...",
                     market_slug, result["up_token_id"][:16], result["down_token_id"][:16])
            return result

        except requests.RequestException as e:
            log.error("Error HTTP resolviendo %s: %s", market_slug, e)
            return None
        except Exception as e:
            log.error("Error resolviendo %s: %s", market_slug, e)
            return None


# =============================================================================
# Cliente de ejecución
# =============================================================================
class PolymarketExecutor:
    """
    Ejecuta órdenes en Polymarket vía py-clob-client.
    Market orders = limit order agresiva con slippage.
    """

    def __init__(self, cfg: dict, dry_run: bool = True):
        self.cfg = cfg
        self.dry_run = dry_run
        self.client = None

        if not dry_run:
            self._init_client()

    def _init_client(self):
        """Inicializa el ClobClient de py-clob-client."""
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=self.cfg["API_KEY"],
                api_secret=self.cfg["API_SECRET"],
                api_passphrase=self.cfg["API_PASSPHRASE"],
            )

            self.client = ClobClient(
                self.cfg["CLOB_HOST"],
                key=self.cfg["PRIVATE_KEY"],
                chain_id=self.cfg["CHAIN_ID"],
                creds=creds,
            )
            log.info("ClobClient inicializado (LIVE)")
        except ImportError:
            log.error("py-clob-client no instalado. Ejecuta: pip install py-clob-client")
            raise
        except Exception as e:
            log.error("Error inicializando ClobClient: %s", e)
            raise

    def buy_shares(self, token_id: str, amount_usdc: float, price: float) -> dict:
        """
        Compra shares.
        Market order = limit agresiva a precio alto para fill inmediato.
        price = ask actual del snapshot (la usamos con slippage como limit).
        """
        if self.dry_run:
            return {"status": "DRY_RUN", "token_id": token_id, "amount": amount_usdc}

        try:
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY

            # Precio limit = ask + slippage (agresivo para fill inmediato)
            limit_price = min(round(price + self.cfg["SLIPPAGE"], 2), 0.99)
            # Size en shares = USDC / price
            size = round(amount_usdc / price, 2)

            order_args = OrderArgs(
                token_id=token_id,
                price=limit_price,
                size=size,
                side=BUY,
            )
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order)
            return {"status": "EXECUTED", "order": resp}

        except Exception as e:
            log.error("Error ejecutando BUY: %s", e)
            return {"status": "ERROR", "error": str(e)}

    def sell_shares(self, token_id: str, size: float, price: float) -> dict:
        """
        Vende shares.
        Market sell = limit agresiva a precio bajo para fill inmediato.
        price = bid actual del snapshot.
        """
        if self.dry_run:
            return {"status": "DRY_RUN", "token_id": token_id, "size": size}

        try:
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import SELL

            # Precio limit = bid - slippage (agresivo para fill inmediato)
            limit_price = max(round(price - self.cfg["SLIPPAGE"], 2), 0.01)

            order_args = OrderArgs(
                token_id=token_id,
                price=limit_price,
                size=round(size, 2),
                side=SELL,
            )
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order)
            return {"status": "EXECUTED", "order": resp}

        except Exception as e:
            log.error("Error ejecutando SELL: %s", e)
            return {"status": "ERROR", "error": str(e)}


# =============================================================================
# Log de ejecuciones
# =============================================================================
EXEC_COLUMNS = [
    "ts_executed", "market_slug", "signal", "token_id",
    "side", "amount", "entry_ask", "exit_bid",
    "status", "dry_run", "detail",
]


class ExecutionLogger:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            pd.DataFrame(columns=EXEC_COLUMNS).to_csv(path, index=False)

    def log_execution(self, record: dict):
        row = {col: record.get(col, "") for col in EXEC_COLUMNS}
        pd.DataFrame([row]).to_csv(self.path, mode="a", header=False, index=False)


# =============================================================================
# Lector incremental de señales
# =============================================================================
class SignalReader:
    """Lee trades_5m.csv incrementalmente, retorna solo filas nuevas."""

    def __init__(self, path: str):
        self.path = path
        self._last_count = 0

        # Contar filas ya existentes al arrancar (no re-ejecutar señales viejas)
        if os.path.exists(path):
            try:
                existing = pd.read_csv(path)
                self._last_count = len(existing)
                log.info("Señales existentes: %d (se ignorarán)", self._last_count)
            except Exception:
                pass

    def read_new(self) -> pd.DataFrame:
        """Retorna DataFrame con filas nuevas desde la última lectura."""
        if not os.path.exists(self.path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.path)
        except Exception:
            return pd.DataFrame()

        if len(df) <= self._last_count:
            return pd.DataFrame()

        new_rows = df.iloc[self._last_count:]
        self._last_count = len(df)
        return new_rows


# =============================================================================
# Tracker de posiciones (para saber cuántas shares vender en EXIT)
# =============================================================================
class PositionTracker:
    """
    Registra posiciones abiertas: market_slug → {direction, token_id, shares, entry_ask}
    Para saber cuántas shares vender cuando llega un EXIT.
    """

    def __init__(self):
        # market_slug → {direction, token_id, shares, entry_ask}
        self._positions: dict[str, dict] = {}

    def open(self, mkt: str, direction: str, token_id: str, shares: float, entry_ask: float):
        self._positions[mkt] = {
            "direction": direction,
            "token_id": token_id,
            "shares": shares,
            "entry_ask": entry_ask,
        }

    def get(self, mkt: str) -> dict | None:
        return self._positions.get(mkt)

    def close(self, mkt: str):
        self._positions.pop(mkt, None)

    @property
    def n_open(self) -> int:
        return len(self._positions)


# =============================================================================
# Loop principal
# =============================================================================
def run(args):
    dry_run = not args.live

    log.info("=" * 60)
    log.info("  Order Executor — Polymarket BTC")
    log.info("  Modo: %s", "🔴 LIVE" if not dry_run else "🟡 DRY-RUN")
    log.info("=" * 60)

    cfg = CFG.copy()
    if args.stake:
        cfg["STAKE"] = args.stake

    # Componentes
    resolver = MarketResolver()
    executor = PolymarketExecutor(cfg, dry_run=dry_run)
    sig_reader = SignalReader(cfg["SIGNALS_PATH"])
    exec_log = ExecutionLogger(cfg["EXEC_LOG_PATH"])
    positions = PositionTracker()

    log.info("Señales: %s", cfg["SIGNALS_PATH"])
    log.info("Ejecuciones: %s", cfg["EXEC_LOG_PATH"])
    log.info("Stake: %.2f USDC", cfg["STAKE"])

    n_executed = 0

    try:
        while True:
            new_signals = sig_reader.read_new()

            if new_signals.empty:
                time.sleep(args.poll)
                continue

            for _, row in new_signals.iterrows():
                mkt = row["market_slug"]
                signal = row["signal"]
                ts_now = datetime.now(timezone.utc).isoformat()

                # ─────────────────────────────────────────────────────────
                # ENTRY: LONG_UP / LONG_DOWN
                # ─────────────────────────────────────────────────────────
                if signal in ("LONG_UP", "LONG_DOWN"):
                    direction = "UP" if signal == "LONG_UP" else "DOWN"

                    # Resolver token_id
                    market_info = resolver.resolve(mkt)
                    if market_info is None:
                        log.warning("⚠️ No se pudo resolver %s — skip", mkt)
                        exec_log.log_execution({
                            "ts_executed": ts_now, "market_slug": mkt,
                            "signal": signal, "side": "BUY",
                            "status": "SKIP_NO_TOKEN", "dry_run": dry_run,
                        })
                        continue

                    token_id = market_info[f"{direction.lower()}_token_id"]
                    entry_ask = float(row.get("entry_ask", 0))
                    stake = cfg["STAKE"]

                    # Shares a comprar = STAKE / ask
                    shares = stake / entry_ask if entry_ask > 0 else 0

                    log.info("📤 %s %s │ %s │ ask=%.4f │ shares=%.1f │ $%.2f",
                             "BUY" if not dry_run else "DRY_BUY",
                             direction, mkt, entry_ask, shares, stake)

                    result = executor.buy_shares(token_id, stake, entry_ask)

                    # Registrar posición
                    positions.open(mkt, direction, token_id, shares, entry_ask)

                    exec_log.log_execution({
                        "ts_executed": ts_now, "market_slug": mkt,
                        "signal": signal, "token_id": token_id,
                        "side": "BUY", "amount": stake,
                        "entry_ask": entry_ask,
                        "status": result["status"], "dry_run": dry_run,
                        "detail": json.dumps(result.get("order", result.get("error", "")))[:200],
                    })
                    n_executed += 1

                # ─────────────────────────────────────────────────────────
                # EXIT: EXIT_UP / EXIT_DOWN
                # ─────────────────────────────────────────────────────────
                elif signal in ("EXIT_UP", "EXIT_DOWN"):
                    pos = positions.get(mkt)

                    if pos is None:
                        # No tenemos la posición registrada (puede pasar si
                        # el executor se reinició). Intentar resolver igualmente.
                        log.warning("⚠️ EXIT sin posición registrada para %s", mkt)
                        market_info = resolver.resolve(mkt)
                        if market_info is None:
                            exec_log.log_execution({
                                "ts_executed": ts_now, "market_slug": mkt,
                                "signal": signal, "side": "SELL",
                                "status": "SKIP_NO_POSITION", "dry_run": dry_run,
                            })
                            continue

                        direction = "UP" if signal == "EXIT_UP" else "DOWN"
                        token_id = market_info[f"{direction.lower()}_token_id"]
                        # Sin info de shares → estimar con entry_ask del CSV
                        entry_ask = float(row.get("entry_ask", 0.5))
                        shares = cfg["STAKE"] / entry_ask if entry_ask > 0 else 0
                    else:
                        token_id = pos["token_id"]
                        shares = pos["shares"]
                        entry_ask = pos["entry_ask"]

                    exit_bid = float(row.get("exit_bid", 0))

                    log.info("📥 %s %s │ %s │ bid=%.4f │ shares=%.1f │ entry_ask=%.4f",
                             "SELL" if not dry_run else "DRY_SELL",
                             signal.replace("EXIT_", ""), mkt,
                             exit_bid, shares, entry_ask)

                    result = executor.sell_shares(token_id, shares, exit_bid)

                    positions.close(mkt)

                    exec_log.log_execution({
                        "ts_executed": ts_now, "market_slug": mkt,
                        "signal": signal, "token_id": token_id,
                        "side": "SELL", "amount": shares,
                        "entry_ask": entry_ask, "exit_bid": exit_bid,
                        "status": result["status"], "dry_run": dry_run,
                        "detail": json.dumps(result.get("order", result.get("error", "")))[:200],
                    })
                    n_executed += 1

            log.info("📊 Ejecutadas: %d │ Posiciones abiertas: %d",
                     n_executed, positions.n_open)

            time.sleep(args.poll)

    except KeyboardInterrupt:
        log.info("Detenido. %d órdenes ejecutadas, %d posiciones abiertas.",
                 n_executed, positions.n_open)
        if positions.n_open > 0:
            log.warning("⚠️ Hay %d posiciones abiertas que se resolverán automáticamente "
                        "cuando el mercado cierre.", positions.n_open)


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Order Executor — Polymarket BTC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Lee trades_5m.csv y ejecuta señales en Polymarket.

Por defecto arranca en DRY-RUN (no ejecuta órdenes reales).
Usa --live para ejecución real.

Señales:
  LONG_UP / LONG_DOWN → comprar shares (market order)
  EXIT_UP / EXIT_DOWN → vender shares (market order)

Ficheros:
  trades_5m.csv       ← señales del signal_engine.py
  executed_orders.csv → log de órdenes ejecutadas
  market_cache.json   → cache de market_slug → token_id
        """,
    )
    p.add_argument("--live", action="store_true",
                   help="Ejecución REAL (por defecto: dry-run)")
    p.add_argument("--stake", type=float, default=None,
                   help="Override del stake en USDC")
    p.add_argument("--poll", type=float, default=1.0,
                   help="Intervalo de polling en segundos")

    args = p.parse_args()

    if args.live:
        log.warning("🔴 MODO LIVE — Las órdenes se ejecutarán con dinero real")
        log.warning("   Tienes 5 segundos para cancelar (Ctrl+C)...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Cancelado.")
            return

    run(args)


if __name__ == "__main__":
    main()
