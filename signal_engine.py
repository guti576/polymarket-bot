#!/usr/bin/env python3
"""
signal_engine.py — Motor de señales Polymarket BTC · Strong Momentum

Estrategia 11_Strong_Momentum con umbrales AUTO-CALIBRADOS:
    Mantiene una ventana rolling de btc_return_since_open y recalcula
    los percentiles 75/25 automáticamente. Así los umbrales se adaptan
    si BTC cambia de régimen (más o menos volatilidad).

    Si btc_return_since_open > P75_rolling  → LONG_UP
    Si btc_return_since_open < P25_rolling  → LONG_DOWN
    En otro caso                            → NO_TRADE

Sin ML. Solo una regla dura basada en momentum fuerte de BTC.

Lee incrementalmente polymarket_dataset_5m.csv, genera señales
LONG_UP / LONG_DOWN / NO_TRADE y las escribe en trades_5m.csv.

Uso:
    python signal_engine.py
    python signal_engine.py --window 500 --percentile 75
    python signal_engine.py --no-autocalibrate --ret-up 0.0005
    python signal_engine.py --poll 2.0 --input live_data.csv

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
    # ── Ventana de operación ─────────────────────────────────────────────
    ENTRY_LO=0.10,
    ENTRY_HI=0.60,

    # ── Umbrales iniciales (se usan antes de tener datos suficientes) ────
    RET_THRESHOLD_UP=0.00035,
    RET_THRESHOLD_DOWN=-0.00035,

    # ── Auto-calibración ─────────────────────────────────────────────────
    AUTOCALIBRATE=True,
    CALIBRATION_WINDOW=500,    # nº de snapshots en la ventana rolling
    CALIBRATION_MIN=50,        # mínimo de muestras para empezar a calibrar
    CALIBRATION_PERCENTILE=75, # percentil para el umbral (75 → P75 y P25)

    # ── Apuesta ──────────────────────────────────────────────────────────
    STAKE=10.0,
)


def load_config(config_path: str | None) -> dict:
    """Carga config desde JSON si existe, sino usa defaults."""
    cfg = DEFAULT_CFG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
        log.info("Config cargada desde %s", config_path)
    return cfg


# =============================================================================
# Auto-calibrador de umbrales
# =============================================================================
class ThresholdCalibrator:
    """
    Mantiene una ventana rolling de btc_return_since_open y recalcula
    los percentiles P75/P25 automáticamente.

    Solo alimenta la ventana con snapshots dentro de la zona de operación
    [ENTRY_LO, ENTRY_HI], para ser consistente con cómo se calibraron
    los umbrales en el backtest.

    Comportamiento:
    - Hasta tener CALIBRATION_MIN muestras → usa umbrales iniciales
    - Después → recalcula P75/P25 con los últimos CALIBRATION_WINDOW valores
    - Loguea cada vez que los umbrales cambian significativamente (>10%)
    """

    def __init__(self, cfg: dict):
        self.window_size = cfg["CALIBRATION_WINDOW"]
        self.min_samples = cfg["CALIBRATION_MIN"]
        self.percentile = cfg["CALIBRATION_PERCENTILE"]
        self.enabled = cfg["AUTOCALIBRATE"]

        # Ventana rolling (deque con maxlen = auto-descarta los viejos)
        self._buffer: deque[float] = deque(maxlen=self.window_size)

        # Umbrales actuales (empiezan con los defaults / config)
        self.threshold_up: float = cfg["RET_THRESHOLD_UP"]
        self.threshold_down: float = cfg["RET_THRESHOLD_DOWN"]

        # Para detectar cambios y loguear
        self._last_logged_up: float = self.threshold_up
        self._last_logged_down: float = self.threshold_down

        if self.enabled:
            log.info("Auto-calibración ACTIVADA: window=%d, min=%d, percentil=%d",
                     self.window_size, self.min_samples, self.percentile)
        else:
            log.info("Auto-calibración DESACTIVADA: umbrales fijos UP=%.6f DOWN=%.6f",
                     self.threshold_up, self.threshold_down)

    def feed(self, df: pd.DataFrame, entry_lo: float, entry_hi: float):
        """
        Alimenta la ventana con valores de btc_return_since_open de
        snapshots dentro de la zona de operación.
        """
        if not self.enabled:
            return

        mask = (
            (df["market_progress"] >= entry_lo) &
            (df["market_progress"] <= entry_hi) &
            (df["btc_return_since_open"].notna())
        )
        new_values = df.loc[mask, "btc_return_since_open"].values

        for v in new_values:
            self._buffer.append(float(v))

        # Recalcular si tenemos suficientes muestras
        if len(self._buffer) >= self.min_samples:
            arr = np.array(self._buffer)
            new_up = float(np.percentile(arr, self.percentile))
            new_down = float(np.percentile(arr, 100 - self.percentile))

            # Loguear si cambio significativo (>10% relativo)
            if self._last_logged_up != 0:
                change_up = abs(new_up - self._last_logged_up) / (abs(self._last_logged_up) + 1e-12)
                change_down = abs(new_down - self._last_logged_down) / (abs(self._last_logged_down) + 1e-12)
                if change_up > 0.10 or change_down > 0.10:
                    log.info(
                        "📐 Umbrales recalibrados: UP=%.6f→%.6f (%.1f%%)  "
                        "DOWN=%.6f→%.6f (%.1f%%)  [%d muestras]",
                        self._last_logged_up, new_up, change_up * 100,
                        self._last_logged_down, new_down, change_down * 100,
                        len(self._buffer),
                    )
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
# Columnas necesarias del CSV
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza mínima."""
    df["market_progress"] = df["market_progress"].clip(0.0, 1.0)
    return df


# =============================================================================
# Estrategia: Strong Momentum
# =============================================================================
def evaluate_signals(df: pd.DataFrame, cfg: dict, calibrator: ThresholdCalibrator) -> pd.DataFrame:
    """
    Evalúa cada fila con la regla 11_Strong_Momentum.
    Los umbrales vienen del calibrator (auto-calibrados o fijos).
    """
    df = df.copy()

    lo = cfg["ENTRY_LO"]
    hi = cfg["ENTRY_HI"]
    ret_up = calibrator.threshold_up
    ret_down = calibrator.threshold_down

    signals = []
    for _, row in df.iterrows():
        prog = row["market_progress"]
        if prog < lo or prog > hi:
            signals.append("NO_TRADE")
            continue

        ret = row.get("btc_return_since_open", np.nan)
        if pd.isna(ret):
            signals.append("NO_TRADE")
            continue

        if ret > ret_up:
            ask = row.get("up_ask_p_1", np.nan)
            if pd.isna(ask) or ask <= 0 or ask >= 1.0:
                signals.append("NO_TRADE")
            else:
                signals.append("LONG_UP")
        elif ret < ret_down:
            ask = row.get("down_ask_p_1", np.nan)
            if pd.isna(ask) or ask <= 0 or ask >= 1.0:
                signals.append("NO_TRADE")
            else:
                signals.append("LONG_DOWN")
        else:
            signals.append("NO_TRADE")

    df["signal"] = signals
    return df


# =============================================================================
# Lector incremental de CSV
# =============================================================================
class IncrementalCSVReader:
    """Lee un CSV que crece por append. Solo parsea filas nuevas."""

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
            else:
                self._byte_offset += len(raw.encode("utf-8"))

            chunk = self._header + raw

        available = self._header_cols
        if self.usecols:
            usecols = [c for c in self.usecols if c in available]
        else:
            usecols = None

        df = pd.read_csv(StringIO(chunk), parse_dates=["timestamp"], usecols=usecols)
        return df


# =============================================================================
# Writer de señales
# =============================================================================
OUTPUT_COLUMNS = [
    "ts_data", "ts_processed", "delay_ms",
    "market_slug", "signal",
    "btc_return", "threshold_up", "threshold_down", "calibrated",
    "market_progress",
    "up_ask_p_1", "down_ask_p_1", "up_bid_p_1", "down_bid_p_1",
    "entry_ask", "potential_payout",
]


class SignalWriter:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(path, index=False)
            log.info("Creado %s", path)

    def append(self, records: list[dict]):
        if not records:
            return
        df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
        df.to_csv(self.path, mode="a", header=False, index=False)


# =============================================================================
# Tracker de mercados
# =============================================================================
class MarketTracker:
    """1 trade por mercado."""

    def __init__(self, trades_path: str):
        self._active: dict[str, str] = {}
        if os.path.exists(trades_path):
            try:
                existing = pd.read_csv(trades_path, usecols=["market_slug", "signal"])
                trades = existing[existing["signal"].isin(["LONG_UP", "LONG_DOWN"])]
                first = trades.drop_duplicates(subset="market_slug", keep="first")
                self._active = dict(zip(first["market_slug"], first["signal"]))
                log.info("Recuperados %d mercados con trade activo", len(self._active))
            except Exception:
                pass

    def is_active(self, market_slug: str) -> bool:
        return market_slug in self._active

    def get_signal(self, market_slug: str) -> str:
        return self._active.get(market_slug, "")

    def mark(self, market_slug: str, signal: str):
        self._active[market_slug] = signal

    @property
    def count(self) -> int:
        return len(self._active)


# =============================================================================
# Loop principal
# =============================================================================
def run(args):
    log.info("=" * 60)
    log.info("  Signal Engine — 11_Strong_Momentum (auto-calibrated)")
    log.info("=" * 60)

    cfg = load_config(args.config)

    # CLI overrides
    overrides = {
        "ENTRY_LO": args.entry_lo,
        "ENTRY_HI": args.entry_hi,
        "RET_THRESHOLD_UP": args.ret_up,
        "RET_THRESHOLD_DOWN": args.ret_down,
        "CALIBRATION_WINDOW": args.window,
        "CALIBRATION_PERCENTILE": args.percentile,
    }
    for key, val in overrides.items():
        if val is not None:
            old = cfg.get(key, "N/A")
            cfg[key] = val
            log.info("  Override: %s = %s (was: %s)", key, val, old)

    if args.no_autocalibrate:
        cfg["AUTOCALIBRATE"] = False

    # Calibrador de umbrales
    calibrator = ThresholdCalibrator(cfg)

    log.info("Estrategia: ENTRY=[%.2f, %.2f]", cfg["ENTRY_LO"], cfg["ENTRY_HI"])

    # ── Pre-cargar calibrador con datos históricos del CSV ───────────────
    # Si el CSV ya existe y tiene datos, leemos las últimas N filas para
    # que el calibrador arranque ya calibrado (sin warmup).
    if calibrator.enabled and os.path.exists(args.input):
        try:
            # Leer solo las columnas que necesitamos para calibrar
            hist_cols = ["market_progress", "btc_return_since_open"]
            # Leemos todo y nos quedamos con la cola — para CSVs grandes
            # se podría optimizar, pero con ~100K filas es instantáneo
            df_hist = pd.read_csv(args.input, usecols=hist_cols)
            # Quedarnos con las últimas CALIBRATION_WINDOW×2 filas
            # (×2 porque solo una fracción estará en la ventana de operación)
            tail_size = cfg["CALIBRATION_WINDOW"] * 3
            if len(df_hist) > tail_size:
                df_hist = df_hist.tail(tail_size)
            calibrator.feed(df_hist, cfg["ENTRY_LO"], cfg["ENTRY_HI"])
            log.info("📂 Pre-cargados %d datos históricos de %s → calibrador %s (%d muestras, UP=%.6f DOWN=%.6f)",
                     len(df_hist), args.input,
                     "CALIBRADO" if calibrator.is_calibrated else "en warmup",
                     calibrator.sample_count,
                     calibrator.threshold_up, calibrator.threshold_down)
        except Exception as e:
            log.warning("No se pudo pre-cargar histórico: %s (arrancando con umbrales iniciales)", e)
    else:
        log.info("Umbrales iniciales: UP=%.6f  DOWN=%.6f",
                 calibrator.threshold_up, calibrator.threshold_down)

    # I/O
    reader = IncrementalCSVReader(args.input, usecols=COLS_NEEDED)
    writer = SignalWriter(args.output)
    tracker = MarketTracker(args.output)

    log.info("Vigilando %s (poll=%.1fs)", args.input, args.poll)
    log.info("Señales → %s", args.output)

    # ── Timestamp de lanzamiento: ignorar datos anteriores ───────────────
    launch_time = datetime.now(timezone.utc)
    log.info("🚀 Launch time: %s — datos anteriores se usan solo para calibración",
             launch_time.strftime("%Y-%m-%d %H:%M:%S UTC"))

    n_processed = 0
    n_signals = 0
    n_skipped_historical = 0

    try:
        while True:
            df_new = reader.read_new()

            if df_new is None or df_new.empty:
                time.sleep(args.poll)
                continue

            t_start = time.perf_counter()

            df_new = engineer_features(df_new)

            # ── Alimentar el calibrador con los datos nuevos ─────────────
            # (siempre, incluso datos históricos — para mantener umbrales frescos)
            calibrator.feed(df_new, cfg["ENTRY_LO"], cfg["ENTRY_HI"])

            # ── Evaluar señales con umbrales actuales ────────────────────
            df_eval = evaluate_signals(df_new, cfg, calibrator)

            # Construir output — SOLO trades reales (LONG_UP / LONG_DOWN)
            output_batch = []
            for _, row in df_eval.iterrows():
                mkt = row["market_slug"]
                ts_data = pd.Timestamp(row["timestamp"])
                if ts_data.tzinfo is None:
                    ts_data = ts_data.tz_localize("UTC")

                # ── Ignorar datos anteriores al lanzamiento ──────────
                # Se usan para calibración (arriba) pero no generan trades
                if ts_data < launch_time:
                    n_skipped_historical += 1
                    continue

                ts_now = datetime.now(timezone.utc)
                delay_ms = (ts_now - ts_data).total_seconds() * 1000
                raw_signal = row["signal"]

                # Solo nos interesan trades nuevos
                if raw_signal not in ("LONG_UP", "LONG_DOWN"):
                    continue

                # 1 trade por mercado
                if tracker.is_active(mkt):
                    continue

                tracker.mark(mkt, raw_signal)
                n_signals += 1

                entry_ask = float(row["up_ask_p_1"] if raw_signal == "LONG_UP"
                                  else row["down_ask_p_1"])
                payout = 1.0 / entry_ask if entry_ask > 0 else 0

                cal_tag = "auto" if calibrator.is_calibrated else "fixed"
                log.info(
                    "🔔 %s │ %s │ ret=%.6f │ thresh=[%.6f,%.6f] %s │ "
                    "ask=%.4f │ payout=%.2fx │ prog=%.1f%% │ delay=%dms",
                    raw_signal, mkt,
                    float(row.get("btc_return_since_open", 0)),
                    calibrator.threshold_up, calibrator.threshold_down, cal_tag,
                    entry_ask, payout,
                    float(row["market_progress"]) * 100,
                    delay_ms,
                )

                output_batch.append({
                    "ts_data": ts_data.isoformat(),
                    "ts_processed": ts_now.isoformat(),
                    "delay_ms": round(delay_ms, 1),
                    "market_slug": mkt,
                    "signal": raw_signal,
                    "btc_return": round(float(row.get("btc_return_since_open", np.nan)), 8),
                    "threshold_up": round(calibrator.threshold_up, 8),
                    "threshold_down": round(calibrator.threshold_down, 8),
                    "calibrated": calibrator.is_calibrated,
                    "market_progress": round(float(row["market_progress"]), 4),
                    "up_ask_p_1": round(float(row.get("up_ask_p_1", np.nan)), 5),
                    "down_ask_p_1": round(float(row.get("down_ask_p_1", np.nan)), 5),
                    "up_bid_p_1": round(float(row.get("up_bid_p_1", np.nan)), 5),
                    "down_bid_p_1": round(float(row.get("down_bid_p_1", np.nan)), 5),
                    "entry_ask": round(entry_ask, 5),
                    "potential_payout": round(payout, 4),
                })

            writer.append(output_batch)

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            n_processed += len(df_new)

            cal_status = (f"calibrated ({calibrator.sample_count} samples)"
                          if calibrator.is_calibrated
                          else f"warming up ({calibrator.sample_count}/{calibrator.min_samples})")

            log.info(
                "📊 rows=%d (+%d) │ signals=%d │ active=%d │ skipped_hist=%d │ "
                "thresh=[%.6f, %.6f] %s │ %.0fms",
                n_processed, len(df_new), n_signals, tracker.count, n_skipped_historical,
                calibrator.threshold_up, calibrator.threshold_down, cal_status,
                elapsed_ms,
            )

            time.sleep(args.poll)

    except KeyboardInterrupt:
        log.info("Detenido. Total: %d filas procesadas, %d señales emitidas.",
                 n_processed, n_signals)
        log.info("Umbrales finales: UP=%.6f  DOWN=%.6f  (%s, %d muestras)",
                 calibrator.threshold_up, calibrator.threshold_down,
                 "calibrado" if calibrator.is_calibrated else "inicial",
                 calibrator.sample_count)


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Signal Engine — 11_Strong_Momentum (auto-calibrated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Estrategia 11_Strong_Momentum con auto-calibración:

  Mantiene una ventana rolling de btc_return_since_open y recalcula
  los percentiles P75/P25 automáticamente. Los umbrales se adaptan
  si BTC cambia de régimen.

  Fase 1 (warmup):  usa umbrales iniciales hasta tener >=50 muestras
  Fase 2 (calibrado): P75/P25 rolling sobre últimos 500 snapshots

Restricciones:
  - Solo opera en market_progress ∈ [ENTRY_LO, ENTRY_HI]
  - 1 trade por mercado (máximo)
  - Hold hasta resolución (sin exit anticipado)

Ejemplos:
  python signal_engine.py                          # auto-calibración ON
  python signal_engine.py --window 1000            # ventana más larga
  python signal_engine.py --percentile 80          # umbrales más estrictos
  python signal_engine.py --no-autocalibrate       # umbrales fijos
  python signal_engine.py --no-autocalibrate --ret-up 0.0005
        """,
    )
    # I/O
    p.add_argument("--input", default="polymarket_dataset_5m.csv",
                   help="CSV de entrada (append-only)")
    p.add_argument("--output", default="trades_5m.csv",
                   help="CSV de salida con señales")
    p.add_argument("--config", default=None,
                   help="JSON con configuración (override de defaults)")
    p.add_argument("--poll", type=float, default=1.0,
                   help="Intervalo de polling en segundos")

    # Estrategia
    s = p.add_argument_group("estrategia")
    s.add_argument("--entry-lo", type=float, default=None,
                   help="market_progress mínimo (default: 0.10)")
    s.add_argument("--entry-hi", type=float, default=None,
                   help="market_progress máximo (default: 0.60)")
    s.add_argument("--ret-up", type=float, default=None,
                   help="Umbral inicial UP (default: 0.00035)")
    s.add_argument("--ret-down", type=float, default=None,
                   help="Umbral inicial DOWN (default: -0.00035)")

    # Auto-calibración
    c = p.add_argument_group("auto-calibración")
    c.add_argument("--no-autocalibrate", action="store_true",
                   help="Desactivar auto-calibración (umbrales fijos)")
    c.add_argument("--window", type=int, default=None,
                   help="Tamaño ventana rolling (default: 500)")
    c.add_argument("--percentile", type=int, default=None,
                   help="Percentil para umbrales (default: 75)")

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
