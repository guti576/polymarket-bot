#!/usr/bin/env python3
"""
signal_engine.py — Polymarket BTC · Strong Momentum + Mom Reversal Exit

Entrada: 11_Strong_Momentum (rolling P75/P25, N=500)
    btc_return_since_open > P75 → LONG_UP
    btc_return_since_open < P25 → LONG_DOWN

Salida: Momentum Reversal
    Si entramos UP  y btc_return cae por debajo de -margin → EXIT_UP
    Si entramos DOWN y btc_return sube por encima de +margin → EXIT_DOWN
    Solo después de min_progress del mercado.

Restricciones:
    - 1 trade por mercado (entra una vez, sale una vez o hold a resolución)
    - Solo opera en market_progress ∈ [ENTRY_LO, ENTRY_HI]
    - No genera trades con datos anteriores al lanzamiento

Señales emitidas:
    LONG_UP / LONG_DOWN   → comprar shares
    EXIT_UP / EXIT_DOWN   → vender shares al bid

Requisitos:
    pip install pandas numpy
"""

import argparse
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("signal_engine")


# =============================================================================
# Configuración
# =============================================================================
DEFAULT_CFG = dict(
    # ── Entrada ──────────────────────────────────────────────────────────
    ENTRY_LO=0.10,
    ENTRY_HI=0.60,
    RET_THRESHOLD_UP=0.00035,
    RET_THRESHOLD_DOWN=-0.00035,

    # ── Auto-calibración (entrada) ───────────────────────────────────────
    AUTOCALIBRATE=True,
    CALIBRATION_WINDOW=500,
    CALIBRATION_MIN=50,
    CALIBRATION_PERCENTILE=75,

    # ── Salida: Mom Reversal ─────────────────────────────────────────────
    EXIT_MARGIN=0.0001,       # btc_return tiene que cruzar cero + margin
    EXIT_MIN_PROGRESS=0.15,   # no salir antes de este % del mercado

    # ── Apuesta ──────────────────────────────────────────────────────────
    STAKE=10.0,
)


def load_config(config_path: str | None) -> dict:
    cfg = DEFAULT_CFG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg.update(json.load(f))
        log.info("Config cargada desde %s", config_path)
    return cfg


# =============================================================================
# Auto-calibrador de umbrales (entrada)
# =============================================================================
class ThresholdCalibrator:
    def __init__(self, cfg: dict):
        self.window_size = cfg["CALIBRATION_WINDOW"]
        self.min_samples = cfg["CALIBRATION_MIN"]
        self.percentile = cfg["CALIBRATION_PERCENTILE"]
        self.enabled = cfg["AUTOCALIBRATE"]
        self._buffer: deque[float] = deque(maxlen=self.window_size)
        self.threshold_up: float = cfg["RET_THRESHOLD_UP"]
        self.threshold_down: float = cfg["RET_THRESHOLD_DOWN"]
        self._last_logged_up = self.threshold_up
        self._last_logged_down = self.threshold_down

    def feed(self, df: pd.DataFrame, entry_lo: float, entry_hi: float):
        if not self.enabled:
            return
        mask = (
            (df["market_progress"] >= entry_lo) &
            (df["market_progress"] <= entry_hi) &
            (df["btc_return_since_open"].notna())
        )
        for v in df.loc[mask, "btc_return_since_open"].values:
            self._buffer.append(float(v))
        if len(self._buffer) >= self.min_samples:
            arr = np.array(self._buffer)
            new_up = float(np.percentile(arr, self.percentile))
            new_down = float(np.percentile(arr, 100 - self.percentile))
            if self._last_logged_up != 0:
                ch_up = abs(new_up - self._last_logged_up) / (abs(self._last_logged_up) + 1e-12)
                ch_dn = abs(new_down - self._last_logged_down) / (abs(self._last_logged_down) + 1e-12)
                if ch_up > 0.10 or ch_dn > 0.10:
                    log.info("📐 Recalibrado: UP=%.6f→%.6f  DOWN=%.6f→%.6f  [%d]",
                             self._last_logged_up, new_up, self._last_logged_down, new_down,
                             len(self._buffer))
                    self._last_logged_up = new_up
                    self._last_logged_down = new_down
            self.threshold_up = new_up
            self.threshold_down = new_down

    @property
    def is_calibrated(self) -> bool:
        return len(self._buffer) >= self.min_samples

    @property
    def sample_count(self) -> int:
        return len(self._buffer)


# =============================================================================
# Market Tracker — 3 estados: unseen → open → closed
# =============================================================================
class MarketTracker:
    """
    Tres estados por mercado:
        unseen  → no se ha operado
        open    → posición abierta (esperando exit o resolución)
        closed  → ya se salió o el mercado terminó, no se vuelve a operar

    Cuando se abre una posición, guarda dirección y entry_ask para
    poder evaluar la salida.
    """

    def __init__(self, trades_path: str):
        # market_slug → {'state': 'open'|'closed', 'direction': str, 'entry_ask': float}
        self._markets: dict[str, dict] = {}

        # Reconstruir estado de runs anteriores
        if os.path.exists(trades_path):
            try:
                existing = pd.read_csv(trades_path, usecols=["market_slug", "signal"])
                for _, row in existing.iterrows():
                    mkt = row["market_slug"]
                    sig = row["signal"]
                    if sig in ("LONG_UP", "LONG_DOWN"):
                        if mkt not in self._markets:
                            direction = "UP" if sig == "LONG_UP" else "DOWN"
                            self._markets[mkt] = {"state": "open", "direction": direction,
                                                   "entry_ask": np.nan}
                    elif sig in ("EXIT_UP", "EXIT_DOWN"):
                        if mkt in self._markets:
                            self._markets[mkt]["state"] = "closed"
                n_open = sum(1 for v in self._markets.values() if v["state"] == "open")
                n_closed = sum(1 for v in self._markets.values() if v["state"] == "closed")
                log.info("Recuperados %d mercados (%d open, %d closed)", len(self._markets), n_open, n_closed)
            except Exception:
                pass

    def is_unseen(self, mkt: str) -> bool:
        return mkt not in self._markets

    def is_open(self, mkt: str) -> bool:
        return self._markets.get(mkt, {}).get("state") == "open"

    def get_open_positions(self) -> dict[str, dict]:
        """Retorna {market_slug: {direction, entry_ask}} de posiciones abiertas."""
        return {mkt: info for mkt, info in self._markets.items() if info["state"] == "open"}

    def open_position(self, mkt: str, direction: str, entry_ask: float):
        self._markets[mkt] = {"state": "open", "direction": direction, "entry_ask": entry_ask}

    def close_position(self, mkt: str):
        if mkt in self._markets:
            self._markets[mkt]["state"] = "closed"

    @property
    def n_open(self) -> int:
        return sum(1 for v in self._markets.values() if v["state"] == "open")

    @property
    def n_total(self) -> int:
        return len(self._markets)


# =============================================================================
# CSV reader / writer
# =============================================================================
COLS_NEEDED = [
    "timestamp", "market_slug", "market_progress",
    "btc_return_since_open",
    "up_ask_p_1", "down_ask_p_1",
    "up_bid_p_1", "down_bid_p_1",
    "up_ask_s_1", "down_ask_s_1",
    "up_win_net", "down_win_net",
    "up_loss_net", "down_loss_net",
]

OUTPUT_COLUMNS = [
    "ts_data", "ts_processed", "delay_ms",
    "market_slug", "signal",
    "btc_return", "market_progress",
    "entry_ask", "exit_bid",
    "threshold_up", "threshold_down", "calibrated",
    "up_ask_p_1", "down_ask_p_1", "up_bid_p_1", "down_bid_p_1",
]


class IncrementalCSVReader:
    def __init__(self, path: str, usecols: list[str] | None = None):
        self.path = path
        self.usecols = usecols
        self._byte_offset: int = 0
        self._header: str | None = None
        self._header_cols: list[str] | None = None

    def read_new(self) -> pd.DataFrame | None:
        try:
            file_size = os.path.getsize(self.path)
        except FileNotFoundError:
            return None
        if file_size <= self._byte_offset:
            return None
        with open(self.path, "r", encoding="utf-8") as f:
            if self._header is None:
                self._header = f.readline()
                self._header_cols = [c.strip() for c in self._header.strip().split(",")]
                self._byte_offset = f.tell()
                remaining = f.read()
                if not remaining.strip():
                    return None
                raw = remaining
            else:
                f.seek(self._byte_offset)
                raw = f.read()
                if not raw.strip():
                    return None
            if not raw.endswith("\n"):
                last_nl = raw.rfind("\n")
                if last_nl == -1:
                    return None
                raw = raw[: last_nl + 1]
            self._byte_offset += len(raw.encode("utf-8"))
            chunk = self._header + raw
        usecols = ([c for c in self.usecols if c in self._header_cols]
                    if self.usecols else None)
        return pd.read_csv(StringIO(chunk), parse_dates=["timestamp"], usecols=usecols)


class SignalWriter:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(path, index=False)
            log.info("Creado %s", path)

    def append(self, records: list[dict]):
        if not records:
            return
        pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(
            self.path, mode="a", header=False, index=False)


# =============================================================================
# Loop principal
# =============================================================================
def run(args):
    log.info("=" * 60)
    log.info("  Signal Engine — Strong Momentum + Mom Reversal Exit")
    log.info("=" * 60)

    cfg = load_config(args.config)

    # CLI overrides
    for key, val in {
        "ENTRY_LO": args.entry_lo, "ENTRY_HI": args.entry_hi,
        "RET_THRESHOLD_UP": args.ret_up, "RET_THRESHOLD_DOWN": args.ret_down,
        "CALIBRATION_WINDOW": args.window, "CALIBRATION_PERCENTILE": args.percentile,
    }.items():
        if val is not None:
            old = cfg.get(key, "N/A")
            cfg[key] = val
            log.info("  Override: %s = %s (was: %s)", key, val, old)
    if args.no_autocalibrate:
        cfg["AUTOCALIBRATE"] = False

    # Calibrador
    calibrator = ThresholdCalibrator(cfg)

    # Pre-cargar histórico para calibración
    if calibrator.enabled and os.path.exists(args.input):
        try:
            hist = pd.read_csv(args.input,
                               usecols=["market_progress", "btc_return_since_open"])
            tail = cfg["CALIBRATION_WINDOW"] * 3
            if len(hist) > tail:
                hist = hist.tail(tail)
            calibrator.feed(hist, cfg["ENTRY_LO"], cfg["ENTRY_HI"])
            log.info("📂 Pre-cargados %d históricos → %s (%d muestras, UP=%.6f DOWN=%.6f)",
                     len(hist),
                     "CALIBRADO" if calibrator.is_calibrated else "warmup",
                     calibrator.sample_count,
                     calibrator.threshold_up, calibrator.threshold_down)
        except Exception as e:
            log.warning("No se pudo pre-cargar: %s", e)

    # I/O
    reader = IncrementalCSVReader(args.input, usecols=COLS_NEEDED)
    writer = SignalWriter(args.output)
    tracker = MarketTracker(args.output)

    # Parámetros de exit
    exit_margin = cfg["EXIT_MARGIN"]
    exit_min_prog = cfg["EXIT_MIN_PROGRESS"]

    log.info("Entrada: ENTRY=[%.2f,%.2f]  thresh=[%.6f,%.6f]",
             cfg["ENTRY_LO"], cfg["ENTRY_HI"],
             calibrator.threshold_up, calibrator.threshold_down)
    log.info("Salida:  EXIT_MARGIN=%.6f  EXIT_MIN_PROGRESS=%.2f", exit_margin, exit_min_prog)

    launch_time = datetime.now(timezone.utc)
    log.info("🚀 Launch: %s", launch_time.strftime("%Y-%m-%d %H:%M:%S UTC"))

    n_processed = 0
    n_entries = 0
    n_exits = 0
    n_skipped = 0

    try:
        while True:
            df_new = reader.read_new()
            if df_new is None or df_new.empty:
                time.sleep(args.poll)
                continue

            t_start = time.perf_counter()
            df_new = df_new.copy()
            df_new["market_progress"] = df_new["market_progress"].clip(0.0, 1.0)

            # ── Siempre: alimentar calibrador ────────────────────────────
            calibrator.feed(df_new, cfg["ENTRY_LO"], cfg["ENTRY_HI"])

            output_batch = []
            lo, hi = cfg["ENTRY_LO"], cfg["ENTRY_HI"]
            open_positions = tracker.get_open_positions()
            has_open = len(open_positions) > 0

            for _, row in df_new.iterrows():
                mkt = row["market_slug"]
                ts_data = pd.Timestamp(row["timestamp"])
                if ts_data.tzinfo is None:
                    ts_data = ts_data.tz_localize("UTC")

                # ── Datos históricos: solo calibración ───────────────────
                if ts_data < launch_time:
                    n_skipped += 1
                    continue

                ts_now = datetime.now(timezone.utc)
                delay_ms = (ts_now - ts_data).total_seconds() * 1000
                ret = row.get("btc_return_since_open", np.nan)
                prog = row["market_progress"]

                # ═════════════════════════════════════════════════════════
                # FASE 1: CHECK EXIT (solo si hay posiciones abiertas)
                # ═════════════════════════════════════════════════════════
                if has_open and mkt in open_positions:
                    pos = open_positions[mkt]
                    direction = pos["direction"]

                    if not pd.isna(ret) and prog >= exit_min_prog:
                        should_exit = False
                        if direction == "UP" and ret < -exit_margin:
                            should_exit = True
                        elif direction == "DOWN" and ret > exit_margin:
                            should_exit = True

                        if should_exit:
                            bid_col = "up_bid_p_1" if direction == "UP" else "down_bid_p_1"
                            bid = row.get(bid_col, np.nan)
                            if not pd.isna(bid) and bid > 0:
                                signal = f"EXIT_{direction}"
                                tracker.close_position(mkt)
                                n_exits += 1

                                # Actualizar open_positions en vivo
                                del open_positions[mkt]
                                has_open = len(open_positions) > 0

                                log.info(
                                    "🔻 %s │ %s │ ret=%.6f │ bid=%.4f │ "
                                    "entry_ask=%.4f │ prog=%.1f%% │ delay=%dms",
                                    signal, mkt, float(ret), float(bid),
                                    pos["entry_ask"], prog * 100, delay_ms,
                                )

                                output_batch.append({
                                    "ts_data": ts_data.isoformat(),
                                    "ts_processed": ts_now.isoformat(),
                                    "delay_ms": round(delay_ms, 1),
                                    "market_slug": mkt,
                                    "signal": signal,
                                    "btc_return": round(float(ret), 8),
                                    "market_progress": round(prog, 4),
                                    "entry_ask": round(pos["entry_ask"], 5),
                                    "exit_bid": round(float(bid), 5),
                                    "threshold_up": round(calibrator.threshold_up, 8),
                                    "threshold_down": round(calibrator.threshold_down, 8),
                                    "calibrated": calibrator.is_calibrated,
                                    "up_ask_p_1": round(float(row.get("up_ask_p_1", np.nan)), 5),
                                    "down_ask_p_1": round(float(row.get("down_ask_p_1", np.nan)), 5),
                                    "up_bid_p_1": round(float(row.get("up_bid_p_1", np.nan)), 5),
                                    "down_bid_p_1": round(float(row.get("down_bid_p_1", np.nan)), 5),
                                })
                                continue  # este snapshot ya procesado

                    continue  # mercado con posición abierta, no salió → next

                # ═════════════════════════════════════════════════════════
                # FASE 2: CHECK ENTRY (solo mercados no vistos)
                # ═════════════════════════════════════════════════════════
                if not tracker.is_unseen(mkt):
                    continue  # mercado cerrado → ignorar

                if prog < lo or prog > hi:
                    continue
                if pd.isna(ret):
                    continue

                direction = None
                if ret > calibrator.threshold_up:
                    ask = row.get("up_ask_p_1", np.nan)
                    if not pd.isna(ask) and 0 < ask < 1:
                        direction = "UP"
                elif ret < calibrator.threshold_down:
                    ask = row.get("down_ask_p_1", np.nan)
                    if not pd.isna(ask) and 0 < ask < 1:
                        direction = "DOWN"

                if direction is None:
                    continue

                entry_ask = float(row["up_ask_p_1"] if direction == "UP" else row["down_ask_p_1"])
                signal = f"LONG_{direction}"

                tracker.open_position(mkt, direction, entry_ask)
                # Actualizar open_positions en vivo
                open_positions[mkt] = {"direction": direction, "entry_ask": entry_ask}
                has_open = True
                n_entries += 1

                payout = 1.0 / entry_ask if entry_ask > 0 else 0
                cal_tag = "auto" if calibrator.is_calibrated else "fixed"
                log.info(
                    "🔔 %s │ %s │ ret=%.6f │ thresh=[%.6f,%.6f] %s │ "
                    "ask=%.4f │ payout=%.2fx │ prog=%.1f%% │ delay=%dms",
                    signal, mkt, float(ret),
                    calibrator.threshold_up, calibrator.threshold_down, cal_tag,
                    entry_ask, payout, prog * 100, delay_ms,
                )

                output_batch.append({
                    "ts_data": ts_data.isoformat(),
                    "ts_processed": ts_now.isoformat(),
                    "delay_ms": round(delay_ms, 1),
                    "market_slug": mkt,
                    "signal": signal,
                    "btc_return": round(float(ret), 8),
                    "market_progress": round(prog, 4),
                    "entry_ask": round(entry_ask, 5),
                    "exit_bid": "",
                    "threshold_up": round(calibrator.threshold_up, 8),
                    "threshold_down": round(calibrator.threshold_down, 8),
                    "calibrated": calibrator.is_calibrated,
                    "up_ask_p_1": round(float(row.get("up_ask_p_1", np.nan)), 5),
                    "down_ask_p_1": round(float(row.get("down_ask_p_1", np.nan)), 5),
                    "up_bid_p_1": round(float(row.get("up_bid_p_1", np.nan)), 5),
                    "down_bid_p_1": round(float(row.get("down_bid_p_1", np.nan)), 5),
                })

            # ── Escribir batch ───────────────────────────────────────────
            writer.append(output_batch)

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            n_processed += len(df_new)

            cal_st = (f"cal({calibrator.sample_count})"
                      if calibrator.is_calibrated else
                      f"warmup({calibrator.sample_count}/{calibrator.min_samples})")
            log.info(
                "📊 +%d │ entries=%d exits=%d open=%d │ "
                "thresh=[%.6f,%.6f] %s │ %.0fms",
                len(df_new), n_entries, n_exits, tracker.n_open,
                calibrator.threshold_up, calibrator.threshold_down, cal_st,
                elapsed_ms,
            )

            time.sleep(args.poll)

    except KeyboardInterrupt:
        log.info("Detenido. %d entradas, %d salidas, %d open, %d procesadas.",
                 n_entries, n_exits, tracker.n_open, n_processed)


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Signal Engine — Strong Momentum + Mom Reversal Exit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Entrada: Strong Momentum (rolling P75/P25, N=500)
Salida:  Momentum Reversal (margin + min_progress)

Señales: LONG_UP, LONG_DOWN (comprar), EXIT_UP, EXIT_DOWN (vender)
1 trade por mercado. No opera con datos anteriores al lanzamiento.

Ejemplos:
  python signal_engine.py
  python signal_engine.py --window 1000 --percentile 80
  python signal_engine.py --no-autocalibrate --ret-up 0.0005
        """,
    )
    p.add_argument("--input", default="polymarket_dataset_5m.csv")
    p.add_argument("--output", default="trades_5m.csv")
    p.add_argument("--config", default=None)
    p.add_argument("--poll", type=float, default=1.0)

    s = p.add_argument_group("entrada")
    s.add_argument("--entry-lo", type=float, default=None)
    s.add_argument("--entry-hi", type=float, default=None)
    s.add_argument("--ret-up", type=float, default=None)
    s.add_argument("--ret-down", type=float, default=None)

    c = p.add_argument_group("auto-calibración")
    c.add_argument("--no-autocalibrate", action="store_true")
    c.add_argument("--window", type=int, default=None)
    c.add_argument("--percentile", type=int, default=None)

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
