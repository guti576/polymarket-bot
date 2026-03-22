"""Polymarket BTC 5-minute market strategy backtester.

What this script does
--------------------
1. Loads snapshot-level data (one row per market snapshot).
2. Sorts strictly by timestamp within each market to avoid look-ahead bias.
3. Creates a small set of past-only features using shifted/expanding calculations.
4. Defines 5 rule-based strategies.
5. Performs a chronological, market-level train/test split.
6. Grid-searches each strategy on the training markets only.
7. Backtests the best parameters on the test markets.
8. Saves a summary CSV and per-strategy trade logs.

Important assumptions
---------------------
- Each market is traded at most once per strategy.
- The first snapshot that triggers a signal becomes the entry.
- PnL is taken from the precomputed columns:
    up_win_net, up_loss_net, down_win_net, down_loss_net
  which already incorporate fill logic and fees.
- No future information is used for any signal or feature creation.

Usage
-----
python polymarket_strategy_backtest.py --data your_dataset.csv --stake 10 --outdir results

Supported input formats
-----------------------
- .csv
- .parquet
- .feather

Outputs
-------
- <outdir>/strategy_summary.csv
- <outdir>/trades_<strategy_name>.csv
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "timestamp",
    "market_slug",
    "resolution",
    "seconds_left",
    "market_progress",
    "btc_spot",
    "btc_price_market_open",
    "btc_return_since_open",
    "up_ask_p_1",
    "down_ask_p_1",
    "up_fee_bps",
    "down_fee_bps",
    "up_avg_fill",
    "down_avg_fill",
    "up_win_net",
    "up_loss_net",
    "down_win_net",
    "down_loss_net",
    "up_break_even",
    "down_break_even",
    "bid_volume_10",
    "ask_volume_10",
    "orderbook_imbalance",
    "mid_price",
    "spread",
    "ret_1m",
    "ret_3m",
    "ret_5m",
    "ret_10m",
    "ema_3",
    "ema_6",
    "ema_12",
    "ema_24",
    "ema_ratio",
    "price_vs_ema12",
    "volume_1m",
    "volume_3m",
    "volume_5m",
    "volatility_3m",
    "volatility_5m",
    "buy_volume_last200",
    "sell_volume_last200",
    "trade_imbalance",
}

EPS = 1e-12
DEFAULT_STAKE = 10.0
DEFAULT_TRAIN_FRAC = 0.7
MIN_TRAIN_TRADES = 20


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def safe_div(a: pd.Series | np.ndarray | float, b: pd.Series | np.ndarray | float):
    return np.where(np.abs(b) > EPS, np.asarray(a) / np.asarray(b), np.nan)


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".feather":
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    validate_columns(df)

    # Core hygiene.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "market_slug", "resolution"]).copy()
    df["market_slug"] = df["market_slug"].astype(str)
    df["resolution"] = df["resolution"].astype(str).str.lower().str.strip()

    # Coerce numeric columns.
    for col in df.columns:
        if col in {"timestamp", "market_slug", "resolution"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strict chronological order within each market.
    df = df.sort_values(["market_slug", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Basic derived features, strictly past-only.
    df = add_past_only_features(df)

    # Fill some safe missing values for strategy logic.
    fill_cols = [
        "orderbook_imbalance",
        "trade_imbalance",
        "ret_1m",
        "ret_3m",
        "ret_5m",
        "ret_10m",
        "ema_ratio",
        "price_vs_ema12",
        "volatility_3m",
        "volatility_5m",
        "spread_pct",
        "trend_past",
        "flow_past",
        "ret_1m_ewm3",
        "ret_3m_ewm3",
        "ret_5m_ewm3",
        "trade_imbalance_ewm5",
        "orderbook_imbalance_ewm5",
        "volatility_5m_ewm5",
        "spread_pct_ewm5",
    ]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def add_past_only_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("market_slug", group_keys=False)

    def ewm_shift(s: pd.Series, span: int) -> pd.Series:
        return s.shift(1).ewm(span=span, adjust=False).mean()

    df["spread_pct"] = safe_div(df["spread"], df["mid_price"])

    df["ret_1m_ewm3"] = g["ret_1m"].transform(lambda s: ewm_shift(s, 3))
    df["ret_3m_ewm3"] = g["ret_3m"].transform(lambda s: ewm_shift(s, 3))
    df["ret_5m_ewm3"] = g["ret_5m"].transform(lambda s: ewm_shift(s, 3))
    df["trade_imbalance_ewm5"] = g["trade_imbalance"].transform(lambda s: ewm_shift(s, 5))
    df["orderbook_imbalance_ewm5"] = g["orderbook_imbalance"].transform(lambda s: ewm_shift(s, 5))
    df["volatility_5m_ewm5"] = g["volatility_5m"].transform(lambda s: ewm_shift(s, 5))
    df["spread_pct_ewm5"] = g["spread_pct"].transform(lambda s: ewm_shift(s, 5))

    # Composite past-only regime signals.
    df["trend_past"] = (
        0.50 * df["ret_3m_ewm3"].fillna(0.0)
        + 0.30 * df["ret_5m_ewm3"].fillna(0.0)
        + 0.20 * (df["ema_ratio"].fillna(1.0) - 1.0)
    )
    df["flow_past"] = (
        0.55 * df["trade_imbalance_ewm5"].fillna(0.0)
        + 0.45 * df["orderbook_imbalance_ewm5"].fillna(0.0)
    )

    return df


# ---------------------------------------------------------------------
# Signal functions
# ---------------------------------------------------------------------


def signal_momentum(row: pd.Series, p: Dict[str, float]) -> Optional[str]:
    spread_ok = row["spread_pct"] <= p["max_spread_pct"]

    up = (
        row["trend_past"] >= p["min_trend"]
        and row["flow_past"] >= p["min_flow"]
        and row["btc_return_since_open"] >= p["min_open_ret"]
        and row["ema_ratio"] >= p["min_ema_ratio"]
        and row["price_vs_ema12"] >= p["min_price_vs_ema12"]
        and row["up_break_even"] <= p["max_break_even"]
        and spread_ok
    )
    down = (
        row["trend_past"] <= -p["min_trend"]
        and row["flow_past"] <= -p["min_flow"]
        and row["btc_return_since_open"] <= -p["min_open_ret"]
        and row["ema_ratio"] <= (2.0 - p["min_ema_ratio"])
        and row["price_vs_ema12"] <= (2.0 - p["min_price_vs_ema12"])
        and row["down_break_even"] <= p["max_break_even"]
        and spread_ok
    )

    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return None


def signal_mean_reversion(row: pd.Series, p: Dict[str, float]) -> Optional[str]:
    spread_ok = row["spread_pct"] <= p["max_spread_pct"]

    extreme_down = (
        row["btc_return_since_open"] <= -p["extreme_ret"]
        and row["ret_1m"] <= -p["micro_mom"]
        and row["flow_past"] >= p["min_reversal_flow"]
        and row["orderbook_imbalance"] >= p["min_reversal_ob"]
        and row["down_break_even"] <= p["max_break_even"]
        and spread_ok
    )
    extreme_up = (
        row["btc_return_since_open"] >= p["extreme_ret"]
        and row["ret_1m"] >= p["micro_mom"]
        and row["flow_past"] <= -p["min_reversal_flow"]
        and row["orderbook_imbalance"] <= -p["min_reversal_ob"]
        and row["up_break_even"] <= p["max_break_even"]
        and spread_ok
    )

    if extreme_down and not extreme_up:
        return "up"
    if extreme_up and not extreme_down:
        return "down"
    return None


def signal_microstructure(row: pd.Series, p: Dict[str, float]) -> Optional[str]:
    spread_pct = row["spread_pct"]
    vol = row["volatility_5m"]

    score = (
        p["w_flow"] * row["flow_past"]
        + p["w_ob"] * row["orderbook_imbalance"]
        + p["w_trend"] * row["trend_past"]
        + p["w_mom"] * row["ret_3m"]
        - p["w_spread"] * spread_pct
        - p["w_vol"] * vol
    )

    if spread_pct > p["max_spread_pct"]:
        return None

    if score >= p["score_threshold"] and row["up_break_even"] <= p["max_break_even"]:
        return "up"
    if score <= -p["score_threshold"] and row["down_break_even"] <= p["max_break_even"]:
        return "down"
    return None


def signal_value_edge(row: pd.Series, p: Dict[str, float]) -> Optional[str]:
    # Heuristic probability model built only from contemporaneous and past-only features.
    raw_up = (
        p["b0"]
        + p["b_ret"] * row["btc_return_since_open"]
        + p["b_trend"] * row["trend_past"]
        + p["b_flow"] * row["flow_past"]
        + p["b_ema"] * (row["ema_ratio"] - 1.0)
        + p["b_pvema"] * (row["price_vs_ema12"] - 1.0)
        - p["b_vol"] * row["volatility_5m"]
        - p["b_spread"] * row["spread_pct"]
    )
    p_up = float(sigmoid(raw_up))
    p_down = 1.0 - p_up

    ev_up = p_up * row["up_win_net"] + (1.0 - p_up) * row["up_loss_net"]
    ev_down = p_down * row["down_win_net"] + (1.0 - p_down) * row["down_loss_net"]

    if max(ev_up, ev_down) < p["min_edge_usdc"]:
        return None
    if ev_up > ev_down:
        return "up"
    if ev_down > ev_up:
        return "down"
    return None


def signal_regime_switch(row: pd.Series, p: Dict[str, float]) -> Optional[str]:
    spread_ok = row["spread_pct"] <= p["max_spread_pct"]
    low_vol = row["volatility_5m"] <= p["low_vol_thr"]
    high_vol = row["volatility_5m"] >= p["high_vol_thr"]

    # Trend-following in quiet regimes.
    if low_vol and spread_ok:
        if (
            row["trend_past"] >= p["trend_thr"]
            and row["flow_past"] >= p["flow_thr"]
            and row["up_break_even"] <= p["max_break_even"]
        ):
            return "up"
        if (
            row["trend_past"] <= -p["trend_thr"]
            and row["flow_past"] <= -p["flow_thr"]
            and row["down_break_even"] <= p["max_break_even"]
        ):
            return "down"

    # Mean reversion in volatile regimes.
    if high_vol and spread_ok:
        if (
            row["btc_return_since_open"] <= -p["extreme_ret"]
            and row["flow_past"] >= p["flow_thr"]
            and row["up_break_even"] <= p["max_break_even"]
        ):
            return "up"
        if (
            row["btc_return_since_open"] >= p["extreme_ret"]
            and row["flow_past"] <= -p["flow_thr"]
            and row["down_break_even"] <= p["max_break_even"]
        ):
            return "down"

    return None


# ---------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------


@dataclass
class StrategySpec:
    name: str
    signal_fn: Callable[[pd.Series, Dict[str, float]], Optional[str]]
    param_grid: Dict[str, List[float]]


@dataclass
class BacktestResult:
    strategy: str
    params: Dict[str, float]
    n_markets: int
    n_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    median_pnl: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    avg_hold_seconds: float
    trades: pd.DataFrame


def generate_param_grid(param_grid: Dict[str, List[float]]) -> Iterable[Dict[str, float]]:
    keys = list(param_grid.keys())
    for values in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, values))


def backtest_strategy(
    df: pd.DataFrame,
    spec: StrategySpec,
    params: Dict[str, float],
    stake: float,
    market_subset: Optional[set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    trades: List[Dict[str, object]] = []

    if market_subset is not None:
        work = df[df["market_slug"].isin(market_subset)].copy()
    else:
        work = df.copy()

    work = work.sort_values(["market_slug", "timestamp"], kind="mergesort")

    for market_slug, g in work.groupby("market_slug", sort=False):
        g = g.sort_values("timestamp", kind="mergesort")
        resolution = str(g["resolution"].iloc[0]).lower()
        entered = False

        for _, row in g.iterrows():
            side = spec.signal_fn(row, params)
            if side is None:
                continue

            if side == "up":
                pnl = float(row["up_win_net"] if resolution == "up" else row["up_loss_net"])
                win = int(resolution == "up")
                entry_price = float(row["up_avg_fill"])
                break_even = float(row["up_break_even"])
            elif side == "down":
                pnl = float(row["down_win_net"] if resolution == "down" else row["down_loss_net"])
                win = int(resolution == "down")
                entry_price = float(row["down_avg_fill"])
                break_even = float(row["down_break_even"])
            else:
                continue

            trades.append(
                {
                    "market_slug": market_slug,
                    "entry_timestamp": row["timestamp"],
                    "side": side,
                    "resolution": resolution,
                    "entry_price": entry_price,
                    "break_even": break_even,
                    "pnl_usdc": pnl,
                    "return_on_stake": pnl / stake,
                    "win": win,
                    "seconds_left": float(row["seconds_left"]),
                    "market_progress": float(row["market_progress"]),
                    "btc_return_since_open": float(row["btc_return_since_open"]),
                    "trend_past": float(row["trend_past"]),
                    "flow_past": float(row["flow_past"]),
                    "orderbook_imbalance": float(row["orderbook_imbalance"]),
                    "trade_imbalance": float(row["trade_imbalance"]),
                    "volatility_5m": float(row["volatility_5m"]),
                }
            )
            entered = True
            break

        # If no signal appeared, the market is ignored for this strategy.
        _ = entered

    trades_df = pd.DataFrame(trades)
    metrics = summarize_trades(trades_df)
    return trades_df, metrics


def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "median_pnl": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "avg_hold_seconds": 0.0,
        }

    pnl = trades["pnl_usdc"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    cum = pnl.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = float(drawdown.min())

    profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > EPS else float("inf")

    return {
        "n_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "profit_factor": profit_factor,
        "expectancy": float(pnl.mean()),
        "max_drawdown": max_dd,
        "avg_hold_seconds": float(trades["seconds_left"].mean()),
    }


# ---------------------------------------------------------------------
# Train/test split and tuning
# ---------------------------------------------------------------------


def split_markets_chronologically(df: pd.DataFrame, train_frac: float = DEFAULT_TRAIN_FRAC) -> Tuple[set[str], set[str]]:
    first_ts = df.groupby("market_slug", sort=False)["timestamp"].min().sort_values()
    markets = list(first_ts.index)
    if len(markets) < 2:
        raise ValueError("Need at least 2 markets for a train/test split.")

    split_idx = int(max(1, min(len(markets) - 1, round(len(markets) * train_frac))))
    train_markets = set(markets[:split_idx])
    test_markets = set(markets[split_idx:])
    return train_markets, test_markets


def select_best_params(
    df_train: pd.DataFrame,
    spec: StrategySpec,
    stake: float,
    min_trades: int = MIN_TRAIN_TRADES,
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    best_params: Optional[Dict[str, float]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_trades = pd.DataFrame()

    candidates = list(generate_param_grid(spec.param_grid))
    if not candidates:
        raise ValueError(f"No parameter candidates defined for strategy {spec.name}")

    for params in candidates:
        trades_df, metrics = backtest_strategy(df_train, spec, params, stake=stake)
        if metrics["n_trades"] < min_trades:
            continue

        score = metrics["total_pnl"]
        if best_metrics is None or score > best_metrics["total_pnl"]:
            best_params = params
            best_metrics = metrics
            best_trades = trades_df

    # If nothing met the minimum trade count, fall back to the best by pnl.
    if best_metrics is None:
        for params in candidates:
            trades_df, metrics = backtest_strategy(df_train, spec, params, stake=stake)
            score = metrics["total_pnl"]
            if best_metrics is None or score > best_metrics["total_pnl"]:
                best_params = params
                best_metrics = metrics
                best_trades = trades_df

    assert best_params is not None and best_metrics is not None
    return best_params, best_metrics, best_trades


# ---------------------------------------------------------------------
# Strategy catalog
# ---------------------------------------------------------------------


def build_strategies() -> List[StrategySpec]:
    momentum_grid = {
        "min_trend": [0.0000, 0.0005, 0.0010],
        "min_flow": [0.00, 0.05, 0.10],
        "min_open_ret": [0.0000, 0.0005, 0.0010],
        "min_ema_ratio": [1.0000, 1.0005, 1.0010],
        "min_price_vs_ema12": [1.0000, 1.0005, 1.0010],
        "max_spread_pct": [0.0005, 0.0010, 0.0020],
        "max_break_even": [0.52, 0.55, 0.58],
    }

    mean_reversion_grid = {
        "extreme_ret": [0.0008, 0.0015, 0.0025],
        "micro_mom": [0.0000, 0.0002, 0.0005],
        "min_reversal_flow": [0.00, 0.05, 0.10],
        "min_reversal_ob": [0.00, 0.03, 0.06],
        "max_spread_pct": [0.0005, 0.0010, 0.0020],
        "max_break_even": [0.52, 0.55, 0.58],
    }

    microstructure_grid = {
        "w_flow": [0.8, 1.0, 1.2],
        "w_ob": [0.8, 1.0, 1.2],
        "w_trend": [0.4, 0.6, 0.8],
        "w_mom": [0.2, 0.4],
        "w_spread": [0.5, 1.0],
        "w_vol": [0.5, 1.0],
        "score_threshold": [0.02, 0.04, 0.06],
        "max_spread_pct": [0.0005, 0.0010, 0.0020],
        "max_break_even": [0.52, 0.55, 0.58],
    }

    value_grid = {
        "b0": [-0.10, 0.00, 0.10],
        "b_ret": [150.0, 300.0],
        "b_trend": [2.0, 3.0, 4.0],
        "b_flow": [1.5, 2.5, 3.5],
        "b_ema": [2.0, 3.0],
        "b_pvema": [1.0, 2.0],
        "b_vol": [1.0, 2.0],
        "b_spread": [10.0, 20.0],
        "min_edge_usdc": [0.02, 0.05, 0.10],
    }

    regime_grid = {
        "low_vol_thr": [0.0002, 0.0004, 0.0006],
        "high_vol_thr": [0.0006, 0.0010, 0.0015],
        "trend_thr": [0.0000, 0.0004, 0.0008],
        "flow_thr": [0.00, 0.05, 0.10],
        "extreme_ret": [0.0008, 0.0015, 0.0025],
        "max_spread_pct": [0.0005, 0.0010, 0.0020],
        "max_break_even": [0.52, 0.55, 0.58],
    }

    return [
        StrategySpec("momentum_trend", signal_momentum, momentum_grid),
        StrategySpec("mean_reversion", signal_mean_reversion, mean_reversion_grid),
        StrategySpec("microstructure_pressure", signal_microstructure, microstructure_grid),
        StrategySpec("value_vs_break_even", signal_value_edge, value_grid),
        StrategySpec("regime_switch", signal_regime_switch, regime_grid),
    ]


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def build_summary_row(
    strategy_name: str,
    params: Dict[str, float],
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> Dict[str, object]:
    row: Dict[str, object] = {"strategy": strategy_name}
    for k, v in params.items():
        row[f"param__{k}"] = v

    for prefix, metrics in [("train", train_metrics), ("test", test_metrics)]:
        for k, v in metrics.items():
            row[f"{prefix}__{k}"] = v
    return row


def print_summary(df_summary: pd.DataFrame) -> None:
    cols = [
        "strategy",
        "train__n_trades",
        "train__total_pnl",
        "train__win_rate",
        "test__n_trades",
        "test__total_pnl",
        "test__win_rate",
        "test__profit_factor",
        "test__max_drawdown",
    ]
    cols = [c for c in cols if c in df_summary.columns]
    display_df = df_summary[cols].copy()

    with pd.option_context("display.max_columns", None, "display.width", 140):
        print("\nStrategy comparison:\n")
        print(display_df.to_string(index=False))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Polymarket BTC strategies.")
    parser.add_argument("--data", required=True, help="Path to CSV/Parquet/Feather dataset")
    parser.add_argument("--stake", type=float, default=DEFAULT_STAKE, help="Stake size in USDC")
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC, help="Fraction of markets used for training")
    parser.add_argument("--outdir", type=str, default="backtest_results", help="Output directory")
    parser.add_argument("--min-trades", type=int, default=MIN_TRAIN_TRADES, help="Minimum training trades for parameter selection")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    df = preprocess(df)

    # Sanity filters: keep only valid final labels.
    df = df[df["resolution"].isin(["up", "down"])].copy()
    if df.empty:
        raise ValueError("No rows left after filtering resolution to {'up', 'down'}.")

    train_markets, test_markets = split_markets_chronologically(df, train_frac=args.train_frac)
    df_train = df[df["market_slug"].isin(train_markets)].copy()
    df_test = df[df["market_slug"].isin(test_markets)].copy()

    strategies = build_strategies()
    summary_rows: List[Dict[str, object]] = []

    for spec in strategies:
        best_params, train_metrics, train_trades = select_best_params(
            df_train=df_train,
            spec=spec,
            stake=args.stake,
            min_trades=args.min_trades,
        )

        test_trades, test_metrics = backtest_strategy(
            df_test,
            spec,
            best_params,
            stake=args.stake,
        )

        summary_rows.append(build_summary_row(spec.name, best_params, train_metrics, test_metrics))

        # Persist trade logs for inspection.
        train_trades.to_csv(outdir / f"trades_{spec.name}_train.csv", index=False)
        test_trades.to_csv(outdir / f"trades_{spec.name}_test.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["test__total_pnl", "test__profit_factor"], ascending=[False, False]).reset_index(drop=True)
    summary_df.to_csv(outdir / "strategy_summary.csv", index=False)

    print_summary(summary_df)
    print(f"\nSaved summary to: {outdir / 'strategy_summary.csv'}")
    print(f"Saved trade logs to: {outdir}")


if __name__ == "__main__":
    main()
