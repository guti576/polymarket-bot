#!/usr/bin/env python3
"""
signal_engine.py — Motor de señales Polymarket BTC v6

Lee incrementalmente polymarket_dataset_5m.csv, genera señales
LONG_UP / LONG_DOWN / NO_TRADE y las escribe en trades_5m.csv.

Uso:
    python signal_engine.py                        # defaults
    python signal_engine.py --model-dir ./model    # ruta a artefactos
    python signal_engine.py --poll 2.0             # polling cada 2s

Requisitos:
    pip install xgboost pandas numpy

Artefactos (generados por el notebook, sección 5b):
    model/booster.json   — XGBoost serializado
    model/isotonic.pkl   — calibración isotónica
    model/config.json    — FEATURES, TRAIN_MED, CFG
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

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
# Modelo y configuración
# =============================================================================
class Model:
    """Carga los artefactos del notebook y expone predict(df) → p_up."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)

        # Config
        with open(model_dir / "config.json") as f:
            cfg = json.load(f)
        self.features: list[str] = cfg["FEATURES"]
        self.train_med: dict[str, float] = cfg["TRAIN_MED"]
        self.strategy: dict = cfg["CFG"]
        log.info("Config cargada: %d features, strategy=%s", len(self.features), self.strategy)

        # XGBoost
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_dir / "booster.json"))
        log.info("XGBoost booster cargado")

        # Isotonic calibration
        with open(model_dir / "isotonic.pkl", "rb") as f:
            self.iso_cal = pickle.load(f)
        log.info("Calibración isotónica cargada")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Retorna p_up calibrada para cada fila."""
        X = self._to_matrix(df)
        dm = xgb.DMatrix(X, feature_names=self.features)
        p_raw = self.booster.predict(dm)
        p_cal = self.iso_cal.transform(p_raw)
        return p_cal

    def _to_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extrae features y aplica imputación con medianas de train."""
        med = pd.Series(self.train_med)
        X = (
            df[self.features]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(med)
            .values
        )
        return X


# =============================================================================
# Feature engineering (producción)
# =============================================================================
# Columnas mínimas del CSV que necesitamos leer
COLS_NEEDED = [
    "timestamp", "market_slug", "market_progress",
    # BTC
    "ret_1m", "ret_3m", "ret_5m", "ret_10m",
    "btc_return_since_open",
    "volatility_3m", "volatility_5m", "volume_1m",
    # Polymarket L1
    "up_ask_p_1", "down_ask_p_1",
    "up_bid_p_1", "down_bid_p_1",
    "up_bid_s_1", "up_ask_s_1",
    "down_bid_s_1", "down_ask_s_1",
    # Payoffs (para EV en el log)
    "up_win_net", "down_win_net",
    "up_loss_net", "down_loss_net",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las 4 features derivadas, in-place."""
    df["vol_ratio"] = df["volatility_3m"] / (df["volatility_5m"] + 1e-9)
    df["mkt_premium"] = df["up_ask_p_1"] - df["down_ask_p_1"]
    df["up_book_imbalance"] = (
        (df["up_bid_s_1"] - df["up_ask_s_1"])
        / (df["up_bid_s_1"] + df["up_ask_s_1"] + 1e-9)
    )
    df["down_book_imbalance"] = (
        (df["down_bid_s_1"] - df["down_ask_s_1"])
        / (df["down_bid_s_1"] + df["down_ask_s_1"] + 1e-9)
    )
    df["market_progress"] = df["market_progress"].clip(0.0, 1.0)
    return df


# =============================================================================
# Estrategia de señales
# =============================================================================
def evaluate_signals(df: pd.DataFrame, p_up: np.ndarray, cfg: dict) -> pd.DataFrame:
    """
    Evalúa cada fila y retorna un DataFrame con las señales.
    Una fila por snapshot evaluado, con signal = LONG_UP / LONG_DOWN / NO_TRADE.
    """
    df = df.copy()
    df["p_up"] = p_up
    df["p_down"] = 1.0 - p_up

    # Value edge
    df["value_edge_up"] = df["p_up"] - df["up_ask_p_1"]
    df["value_edge_down"] = df["p_down"] - df["down_ask_p_1"]

    # EV
    df["ev_up"] = df["p_up"] * df["up_win_net"] + df["p_down"] * df["up_loss_net"]
    df["ev_down"] = df["p_down"] * df["down_win_net"] + df["p_up"] * df["down_loss_net"]

    lo = cfg["ENTRY_LO"]
    hi = cfg["ENTRY_HI"]
    min_p = cfg["MIN_P"]
    min_ve = cfg["VALUE_EDGE"]

    signals = []
    for _, row in df.iterrows():
        # Filtro de ventana temporal
        prog = row["market_progress"]
        if prog < lo or prog > hi:
            signals.append("NO_TRADE")
            continue

        # Evaluar UP y DOWN
        candidates = []
        for direction, p_col, ask_col, ve_col, ev_col in [
            ("LONG_UP", "p_up", "up_ask_p_1", "value_edge_up", "ev_up"),
            ("LONG_DOWN", "p_down", "down_ask_p_1", "value_edge_down", "ev_down"),
        ]:
            p = float(row[p_col])
            ve = float(row[ve_col])
            if p >= min_p and ve >= min_ve:
                candidates.append((direction, ve))

        if candidates:
            signals.append(max(candidates, key=lambda x: x[1])[0])
        else:
            signals.append("NO_TRADE")

    df["signal"] = signals
    return df


# =============================================================================
# Lector incremental de CSV
# =============================================================================
class IncrementalCSVReader:
    """
    Lee un CSV que crece por append. Solo parsea las filas nuevas
    en cada llamada a read_new(), usando byte offset.
    """

    def __init__(self, path: str, usecols: list[str] | None = None):
        self.path = path
        self.usecols = usecols
        self._byte_offset: int = 0
        self._header: str | None = None
        self._header_cols: list[str] | None = None

    def read_new(self) -> pd.DataFrame | None:
        """Retorna DataFrame con filas nuevas, o None si no hay cambios."""
        try:
            file_size = os.path.getsize(self.path)
        except FileNotFoundError:
            return None

        if file_size <= self._byte_offset:
            return None  # sin datos nuevos

        with open(self.path, "r", encoding="utf-8") as f:
            # Primera lectura: capturar header
            if self._header is None:
                self._header = f.readline()
                self._header_cols = [c.strip() for c in self._header.strip().split(",")]
                self._byte_offset = f.tell()

                # Leer el resto del archivo
                remaining = f.read()
                if not remaining.strip():
                    return None
                chunk = self._header + remaining
                self._byte_offset = f.tell()
            else:
                # Lecturas posteriores: seek al offset y leer solo lo nuevo
                f.seek(self._byte_offset)
                new_data = f.read()
                self._byte_offset = f.tell()

                if not new_data.strip():
                    return None
                chunk = self._header + new_data

        df = pd.read_csv(
            StringIO(chunk),
            parse_dates=["timestamp"],
            usecols=self.usecols,
        )
        return df


# =============================================================================
# Writer de señales (1 fila output por cada fila input)
# =============================================================================
OUTPUT_COLUMNS = [
    "ts_data",           # timestamp del snapshot (del CSV)
    "ts_processed",      # timestamp de procesamiento (para medir delay)
    "delay_ms",          # delay en milisegundos
    "market_slug",
    "signal",            # LONG_UP / LONG_DOWN / NO_TRADE / ACTIVE_TRADE
    "p_up",
    "p_down",
    "value_edge_up",
    "value_edge_down",
    "ev_up",
    "ev_down",
    "market_progress",
    "up_ask_p_1",
    "down_ask_p_1",
    "up_bid_p_1",
    "down_bid_p_1",
]


class SignalWriter:
    """Escribe una fila por cada fila de entrada en trades_5m.csv."""

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
# Tracker de mercados con trade activo
# =============================================================================
class MarketTracker:
    """
    Lleva registro de mercados que ya tienen señal LONG_UP/LONG_DOWN emitida.
    Un solo trade por mercado — consistente con el backtest.
    """

    def __init__(self, trades_path: str):
        self._active: dict[str, str] = {}  # market_slug → signal (LONG_UP/LONG_DOWN)
        # Cargar trades activos de runs anteriores
        if os.path.exists(trades_path):
            try:
                existing = pd.read_csv(trades_path, usecols=["market_slug", "signal"])
                trades = existing[existing["signal"].isin(["LONG_UP", "LONG_DOWN"])]
                # Quedarse con la primera señal por mercado
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
    log.info("  Signal Engine v6 — Polymarket BTC")
    log.info("=" * 60)

    # Cargar modelo
    model = Model(args.model_dir)
    cfg = model.strategy

    # Override de estrategia desde CLI
    overrides = {
        "ENTRY_LO": args.entry_lo,
        "ENTRY_HI": args.entry_hi,
        "MIN_P": args.min_p,
        "VALUE_EDGE": args.value_edge,
        "FLIP_THRESH": args.flip,
    }
    for key, val in overrides.items():
        if val is not None:
            old = cfg[key]
            cfg[key] = val
            log.info("  Override: %s = %.4f (config.json: %.4f)", key, val, old)

    # Solo leer columnas necesarias (velocidad)
    # Filtrar a las que existen en COLS_NEEDED
    reader = IncrementalCSVReader(args.input, usecols=COLS_NEEDED)
    writer = SignalWriter(args.output)
    tracker = MarketTracker(args.output)

    log.info("Vigilando %s (poll=%.1fs)", args.input, args.poll)
    log.info("Señales → %s", args.output)
    log.info("Estrategia: MIN_P=%.2f  VALUE_EDGE=%.2f  ENTRY=[%.2f, %.2f]  FLIP=%.2f",
             cfg["MIN_P"], cfg["VALUE_EDGE"], cfg["ENTRY_LO"], cfg["ENTRY_HI"], cfg["FLIP_THRESH"])

    n_processed = 0
    n_signals = 0

    try:
        while True:
            df_new = reader.read_new()

            if df_new is None or df_new.empty:
                time.sleep(args.poll)
                continue

            t_start = time.perf_counter()

            # Feature engineering
            df_new = engineer_features(df_new)

            # Inferencia sobre TODAS las filas
            p_up = model.predict(df_new)

            # Evaluar señales (sin filtro de tracker — evalúa todo)
            df_eval = evaluate_signals(df_new, p_up, cfg)

            # Construir 1 fila de output por cada fila de input
            output_batch = []
            for _, row in df_eval.iterrows():
                mkt = row["market_slug"]
                ts_now = datetime.now(timezone.utc)
                ts_data = pd.Timestamp(row["timestamp"])
                if ts_data.tzinfo is None:
                    ts_data = ts_data.tz_localize("UTC")
                delay_ms = (ts_now - ts_data).total_seconds() * 1000

                raw_signal = row["signal"]  # LONG_UP / LONG_DOWN / NO_TRADE

                # Si ya hay trade activo para este mercado → ACTIVE_TRADE
                if tracker.is_active(mkt):
                    signal = f"ACTIVE_TRADE:{tracker.get_signal(mkt)}"
                else:
                    signal = raw_signal
                    # Registrar nuevo trade
                    if signal in ("LONG_UP", "LONG_DOWN"):
                        tracker.mark(mkt, signal)
                        n_signals += 1
                        log.info(
                            "🔔 %s │ %s │ p_up=%.3f │ ve_up=%.4f ve_dn=%.4f │ ask=%.3f │ delay=%dms",
                            signal, mkt,
                            float(row["p_up"]),
                            float(row["value_edge_up"]),
                            float(row["value_edge_down"]),
                            float(row["up_ask_p_1"] if signal == "LONG_UP" else row["down_ask_p_1"]),
                            delay_ms,
                        )

                output_batch.append({
                    "ts_data": ts_data.isoformat(),
                    "ts_processed": ts_now.isoformat(),
                    "delay_ms": round(delay_ms, 1),
                    "market_slug": mkt,
                    "signal": signal,
                    "p_up": round(float(row["p_up"]), 5),
                    "p_down": round(float(row["p_down"]), 5),
                    "value_edge_up": round(float(row["value_edge_up"]), 5),
                    "value_edge_down": round(float(row["value_edge_down"]), 5),
                    "ev_up": round(float(row["ev_up"]), 5),
                    "ev_down": round(float(row["ev_down"]), 5),
                    "market_progress": round(float(row["market_progress"]), 4),
                    "up_ask_p_1": round(float(row["up_ask_p_1"]), 5),
                    "down_ask_p_1": round(float(row["down_ask_p_1"]), 5),
                    "up_bid_p_1": round(float(row["up_bid_p_1"]), 5),
                    "down_bid_p_1": round(float(row["down_bid_p_1"]), 5),
                })

            # Escribir todas las filas
            writer.append(output_batch)

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            n_processed += len(df_new)

            log.info(
                "📊 rows=%d (+%d) │ signals=%d │ active=%d │ %.0fms",
                n_processed, len(df_new), n_signals, tracker.count, elapsed_ms,
            )

            time.sleep(args.poll)

    except KeyboardInterrupt:
        log.info("Detenido. Total: %d filas procesadas, %d señales emitidas.", n_processed, n_signals)


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Signal Engine — Polymarket BTC v6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python signal_engine.py
  python signal_engine.py --value-edge 0.08 --min-p 0.70
  python signal_engine.py --entry-lo 0.15 --entry-hi 0.60 --flip 0.20
  python signal_engine.py --poll 0.5 --input live_data.csv

Los parámetros de estrategia usan por defecto los valores del config.json
exportado del notebook. Los flags CLI los sobreescriben sin tocar el archivo.
        """,
    )
    # I/O
    p.add_argument("--model-dir", default="model", help="Directorio con artefactos del modelo")
    p.add_argument("--input", default="polymarket_dataset_5m.csv", help="CSV de entrada (append-only)")
    p.add_argument("--output", default="trades_5m.csv", help="CSV de salida con señales")
    p.add_argument("--poll", type=float, default=1.0, help="Intervalo de polling en segundos")

    # Estrategia (override sobre config.json)
    s = p.add_argument_group("estrategia", "Parámetros de la estrategia (override sobre config.json)")
    s.add_argument("--entry-lo", type=float, default=None, help="market_progress mínimo para entrar (default: config.json)")
    s.add_argument("--entry-hi", type=float, default=None, help="market_progress máximo para entrar (default: config.json)")
    s.add_argument("--min-p", type=float, default=None, help="Probabilidad mínima del modelo para entrar (default: config.json)")
    s.add_argument("--value-edge", type=float, default=None, help="p_modelo - ask mínimo para entrar (default: config.json)")
    s.add_argument("--flip", type=float, default=None, help="Umbral de model-flip para exit (default: config.json)")
    args = p.parse_args()

    # Validar artefactos
    model_dir = Path(args.model_dir)
    for artifact in ["booster.json", "isotonic.pkl", "config.json"]:
        if not (model_dir / artifact).exists():
            log.error("Artefacto no encontrado: %s/%s", model_dir, artifact)
            log.error("Ejecuta la sección 5b del notebook para exportar el modelo.")
            sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
