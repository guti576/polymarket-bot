import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from typing import List, Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ======================== Configuration ========================
STAKE = 100.0                       # Stake used in dataset (must match)
DATA_FILE = "dataset_with_resolution.csv"   # Path to your data
TRAIN_RATIO = 0.8                   # Fraction of data used for parameter tuning
PLOT_FILE = "equity_curves.png"     # Output file for equity curves

# ======================== Data Loading ========================
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    return df

# ======================== Backtesting Core ========================
def simulate_strategy(df: pd.DataFrame, signal_func, **kwargs) -> pd.DataFrame:
    """
    Generic backtest for a given strategy.
    signal_func(row, **kwargs) -> 'UP' or 'DOWN' or None.
    Returns trades DataFrame with timestamps and PnL.
    """
    trades = []
    for market, group in df.groupby('market_slug'):
        group = group.sort_values('timestamp')
        resolution = group['resolution'].iloc[0]
        for _, row in group.iterrows():
            if row.get('seconds_left', 0) <= 0:
                continue
            signal = signal_func(row, **kwargs)
            if signal is None:
                continue
            if signal == 'UP':
                entry = row.get('up_avg_fill')
                win_net = row.get('up_win_net')
                loss_net = row.get('up_loss_net')
                if pd.isna(entry) or pd.isna(win_net) or pd.isna(loss_net):
                    continue
                pnl = win_net if resolution == 'up' else loss_net
            elif signal == 'DOWN':
                entry = row.get('down_avg_fill')
                win_net = row.get('down_win_net')
                loss_net = row.get('down_loss_net')
                if pd.isna(entry) or pd.isna(win_net) or pd.isna(loss_net):
                    continue
                pnl = win_net if resolution == 'down' else loss_net
            else:
                continue
            trades.append({
                'market': market,
                'timestamp': row['timestamp'],
                'signal': signal,
                'entry_price': entry,
                'resolution': resolution,
                'pnl': pnl,
            })
            break  # only first trade per market
    return pd.DataFrame(trades)

def compute_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """Return key metrics including Sharpe ratio."""
    if trades_df.empty:
        return {'total_pnl': 0.0, 'num_trades': 0, 'win_rate': 0.0,
                'avg_pnl': 0.0, 'sharpe': np.nan}
    total_pnl = trades_df['pnl'].sum()
    num_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean()
    avg_pnl = total_pnl / num_trades
    # Approximate annualised Sharpe (trades per day ~ 288 5-min markets)
    if num_trades > 1:
        daily_returns = trades_df.groupby(trades_df['timestamp'].dt.date)['pnl'].sum()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else np.nan
    else:
        sharpe = np.nan
    return {
        'total_pnl': total_pnl,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'sharpe': sharpe
    }

def cumulative_pnl(trades_df: pd.DataFrame) -> pd.Series:
    """Return cumulative PnL indexed by timestamp."""
    if trades_df.empty:
        return pd.Series(dtype=float)
    trades_df = trades_df.sort_values('timestamp')
    cumsum = trades_df['pnl'].cumsum()
    cumsum.index = trades_df['timestamp']
    return cumsum

# ======================== Strategy Definitions ========================
# Each strategy function takes a row and keyword parameters and returns 'UP', 'DOWN', or None.

# 1. Momentum + Break‑Even
def strat1_momentum_break_even(row, ret_col='ret_5m', ret_thresh=0.001, be_thresh=0.5):
    ret = row.get(ret_col)
    if pd.isna(ret):
        return None
    if ret > ret_thresh and row.get('up_break_even', 1) < be_thresh:
        return 'UP'
    if ret < -ret_thresh and row.get('down_break_even', 1) < be_thresh:
        return 'DOWN'
    return None

# 2. Dual Order Book Imbalance
def strat2_dual_orderbook_imb(row, poly_imb_thresh=0.2, binance_imb_thresh=0.2):
    up_bid = row.get('up_bid_s_1')
    up_ask = row.get('up_ask_s_1')
    down_bid = row.get('down_bid_s_1')
    down_ask = row.get('down_ask_s_1')
    if any(pd.isna(x) for x in [up_bid, up_ask, down_bid, down_ask]):
        return None
    if up_bid + up_ask == 0 or down_bid + down_ask == 0:
        return None
    poly_up_imb = (up_bid - up_ask) / (up_bid + up_ask)
    poly_down_imb = (down_bid - down_ask) / (down_bid + down_ask)
    binance_imb = row.get('orderbook_imbalance')
    if pd.isna(binance_imb):
        return None
    if poly_up_imb > poly_down_imb and poly_up_imb > poly_imb_thresh and binance_imb > binance_imb_thresh:
        return 'UP'
    if poly_down_imb > poly_up_imb and poly_down_imb > poly_imb_thresh and binance_imb < -binance_imb_thresh:
        return 'DOWN'
    return None

# 3. Volatility‑Filtered Momentum
def strat3_volatility_filtered_momentum(row, ret_col='ret_5m', ret_thresh=0.001, vol_percentile=50):
    ret = row.get(ret_col)
    vol = row.get('volatility_5m')
    if pd.isna(ret) or pd.isna(vol):
        return None
    # vol_percentile is used as a threshold; we'll compute the actual value during optimisation
    if ret > ret_thresh and vol < vol_percentile:
        return 'UP'
    if ret < -ret_thresh and vol < vol_percentile:
        return 'DOWN'
    return None

# 4. Smart Money Flow
def strat4_smart_money_flow(row, trade_imb_thresh=0.2, volume_thresh=0):
    trade_imb = row.get('trade_imbalance')
    volume = row.get('volume_5m')
    if pd.isna(trade_imb) or pd.isna(volume):
        return None
    if trade_imb > trade_imb_thresh and volume > volume_thresh:
        return 'UP'
    if trade_imb < -trade_imb_thresh and volume > volume_thresh:
        return 'DOWN'
    return None

# 5. Price Extremes (Mean Reversion)
def strat5_price_extremes(row, price_thresh=0.6, ret_thresh=0.002):
    up_price = row.get('up_ask_p_1')
    down_price = row.get('down_ask_p_1')
    ret = row.get('ret_5m')
    if any(pd.isna(x) for x in [up_price, down_price, ret]):
        return None
    if up_price > price_thresh and ret < -ret_thresh:
        return 'DOWN'
    if down_price > price_thresh and ret > ret_thresh:
        return 'UP'
    return None

# 6. EMA Cross + Break‑even
def strat6_ema_cross_be(row, ema_ratio_thresh=1.02, price_vs_ema_thresh=1.01, be_thresh=0.5):
    ema_ratio = row.get('ema_ratio')
    price_vs_ema = row.get('price_vs_ema12')
    if pd.isna(ema_ratio) or pd.isna(price_vs_ema):
        return None
    if ema_ratio > ema_ratio_thresh and price_vs_ema > price_vs_ema_thresh and row.get('up_break_even', 1) < be_thresh:
        return 'UP'
    if ema_ratio < 1/ema_ratio_thresh and price_vs_ema < 1/price_vs_ema_thresh and row.get('down_break_even', 1) < be_thresh:
        return 'DOWN'
    return None

# 7. Multi‑timeframe Momentum
def strat7_multi_timeframe(row, ret_5m_thresh=0.001, ret_1m_thresh=0.0005):
    ret_5m = row.get('ret_5m')
    ret_1m = row.get('ret_1m')
    if pd.isna(ret_5m) or pd.isna(ret_1m):
        return None
    if ret_5m > ret_5m_thresh and ret_1m > ret_1m_thresh:
        return 'UP'
    if ret_5m < -ret_5m_thresh and ret_1m < -ret_1m_thresh:
        return 'DOWN'
    return None

# 8. Polymarket Cumulative Depth Imbalance
def strat8_poly_depth_imbalance(row, depth_imb_thresh=0.2):
    # Compute net bid-ask volume across top 3 levels for UP and DOWN
    up_bid_vol = row.get('up_bid_s_1', 0) + row.get('up_bid_s_2', 0) + row.get('up_bid_s_3', 0)
    up_ask_vol = row.get('up_ask_s_1', 0) + row.get('up_ask_s_2', 0) + row.get('up_ask_s_3', 0)
    down_bid_vol = row.get('down_bid_s_1', 0) + row.get('down_bid_s_2', 0) + row.get('down_bid_s_3', 0)
    down_ask_vol = row.get('down_ask_s_1', 0) + row.get('down_ask_s_2', 0) + row.get('down_ask_s_3', 0)
    if up_bid_vol + up_ask_vol == 0 or down_bid_vol + down_ask_vol == 0:
        return None
    up_imb = (up_bid_vol - up_ask_vol) / (up_bid_vol + up_ask_vol)
    down_imb = (down_bid_vol - down_ask_vol) / (down_bid_vol + down_ask_vol)
    if up_imb > down_imb and up_imb > depth_imb_thresh:
        return 'UP'
    if down_imb > up_imb and down_imb > depth_imb_thresh:
        return 'DOWN'
    return None

# 9. Volatility Adjusted Momentum (dynamic)
def strat9_volatility_adjusted(row, ret_5m_thresh_low=0.001, vol_thresh_low=0.0005,
                               ret_5m_thresh_high=0.002, vol_thresh_high=0.001):
    ret = row.get('ret_5m')
    vol = row.get('volatility_5m')
    if pd.isna(ret) or pd.isna(vol):
        return None
    if (ret > ret_5m_thresh_low and vol < vol_thresh_low) or (ret > ret_5m_thresh_high and vol < vol_thresh_high):
        return 'UP'
    if (ret < -ret_5m_thresh_low and vol < vol_thresh_low) or (ret < -ret_5m_thresh_high and vol < vol_thresh_high):
        return 'DOWN'
    return None

# 10. Break‑even Only
def strat10_break_even_only(row, be_thresh=0.45):
    if row.get('up_break_even', 1) < be_thresh:
        return 'UP'
    if row.get('down_break_even', 1) < be_thresh:
        return 'DOWN'
    return None

# 11. Price vs EMA Reversion
def strat11_price_vs_ema_reversion(row, price_dev_thresh=1.02, be_thresh=0.5):
    price_vs_ema = row.get('price_vs_ema12')
    if pd.isna(price_vs_ema):
        return None
    if price_vs_ema > price_dev_thresh and row.get('up_break_even', 1) < be_thresh:
        return 'DOWN'          # Overbought, mean reversion
    if price_vs_ema < 1/price_dev_thresh and row.get('down_break_even', 1) < be_thresh:
        return 'UP'
    return None

# 12. Trade Flow + Order Book Imbalance
def strat12_trade_flow_ob_imb(row, trade_imb_thresh=0.2, ob_imb_thresh=0.2):
    trade_imb = row.get('trade_imbalance')
    ob_imb = row.get('orderbook_imbalance')
    if pd.isna(trade_imb) or pd.isna(ob_imb):
        return None
    if trade_imb > trade_imb_thresh and ob_imb > ob_imb_thresh:
        return 'UP'
    if trade_imb < -trade_imb_thresh and ob_imb < -ob_imb_thresh:
        return 'DOWN'
    return None

# 13. Market Progress Momentum
def strat13_market_progress_momentum(row, progress_thresh=0.5, ret_thresh=0.001):
    progress = row.get('market_progress')
    ret = row.get('ret_5m')
    if pd.isna(progress) or pd.isna(ret):
        return None
    if progress > progress_thresh and ret > ret_thresh:
        return 'UP'
    if progress > progress_thresh and ret < -ret_thresh:
        return 'DOWN'
    return None

# 14. Spread Filter (low spread only)
def strat14_spread_filter(row, spread_thresh=0.0005, ret_thresh=0.001):
    spread = row.get('spread')
    ret = row.get('ret_5m')
    if pd.isna(spread) or pd.isna(ret):
        return None
    if spread < spread_thresh and ret > ret_thresh:
        return 'UP'
    if spread < spread_thresh and ret < -ret_thresh:
        return 'DOWN'
    return None

# 15. Volume Surge
def strat15_volume_surge(row, volume_percentile=75, ret_thresh=0.001):
    volume = row.get('volume_5m')
    ret = row.get('ret_5m')
    if pd.isna(volume) or pd.isna(ret):
        return None
    # volume_percentile is used as a threshold (actual value computed during optimisation)
    if volume > volume_percentile and ret > ret_thresh:
        return 'UP'
    if volume > volume_percentile and ret < -ret_thresh:
        return 'DOWN'
    return None

# ======================== Parameter Optimization ========================
def optimize_strategy(train_df: pd.DataFrame, strategy_func, param_grid: Dict) -> Dict:
    """Find best parameters on training data (maximising total PnL)."""
    best_params = None
    best_pnl = -np.inf
    for params in ParameterGrid(param_grid):
        trades = simulate_strategy(train_df, strategy_func, **params)
        total_pnl = trades['pnl'].sum() if not trades.empty else 0.0
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_params = params
    return best_params or param_grid

def run_optimized_backtest(df: pd.DataFrame, strategies: List[Tuple[str, callable, Dict]]):
    """Split data, optimize on train, test on test, return test trades and metrics."""
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    results = {}
    for name, strat_func, param_grid in strategies:
        print(f"\nOptimising {name}...")
        best_params = optimize_strategy(train_df, strat_func, param_grid)
        test_trades = simulate_strategy(test_df, strat_func, **best_params)
        metrics = compute_metrics(test_trades)
        results[name] = {
            'best_params': best_params,
            'test_trades': test_trades,
            'metrics': metrics
        }
        print(f"Best params: {best_params}")
        print(f"Test PnL: ${metrics['total_pnl']:.2f} | Trades: {metrics['num_trades']} | Win rate: {metrics['win_rate']:.2%}")
    return results

# ======================== Plotting ========================
def plot_equity_curves(results, output_file):
    """Plot cumulative PnL for all strategies on test set."""
    n_strategies = len(results)
    n_cols = 3
    n_rows = (n_strategies + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        trades = res['test_trades']
        if trades.empty:
            axes[idx].text(0.5, 0.5, 'No trades', ha='center', va='center')
            axes[idx].set_title(name)
            continue
        cum = cumulative_pnl(trades)
        axes[idx].plot(cum.index, cum.values, lw=1.5)
        axes[idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[idx].set_title(f"{name}\nPnL: ${res['metrics']['total_pnl']:.2f} | WR: {res['metrics']['win_rate']:.2%}")
        axes[idx].set_xlabel('Time')
        axes[idx].set_ylabel('Cumulative PnL ($)')
        axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"\nEquity curves saved to {output_file}")

# ======================== Main ========================
def main():
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df)} snapshots from {df['market_slug'].nunique()} markets.")
    print(f"Splitting: train first {TRAIN_RATIO:.0%}, test last {1-TRAIN_RATIO:.0%}")

    # Pre‑compute some global percentiles for parameter grids
    vol_values = df['volatility_5m'].dropna()
    vol_percentiles = np.percentile(vol_values, [30, 50, 70]) if len(vol_values) > 0 else [0.0005, 0.001, 0.002]
    volume_values = df['volume_5m'].dropna()
    volume_percentiles = np.percentile(volume_values, [50, 70, 90]) if len(volume_values) > 0 else [100, 200, 500]

    # Define all strategies and their parameter grids
    strategies = [
        # 1. Momentum + Break‑Even
        ("Momentum + BE", strat1_momentum_break_even,
         {'ret_thresh': np.linspace(0.0005, 0.003, 6), 'be_thresh': np.linspace(0.45, 0.55, 5)}),
        # 2. Dual Order Book Imbalance
        ("Dual OB Imb", strat2_dual_orderbook_imb,
         {'poly_imb_thresh': np.linspace(0.1, 0.4, 4), 'binance_imb_thresh': np.linspace(0.1, 0.4, 4)}),
        # 3. Volatility‑Filtered Momentum
        ("Vol Filtered Mom", strat3_volatility_filtered_momentum,
         {'ret_thresh': np.linspace(0.0005, 0.003, 6), 'vol_percentile': vol_percentiles}),
        # 4. Smart Money Flow
        ("Smart Money Flow", strat4_smart_money_flow,
         {'trade_imb_thresh': np.linspace(0.1, 0.4, 4), 'volume_thresh': volume_percentiles}),
        # 5. Price Extremes
        ("Price Extremes", strat5_price_extremes,
         {'price_thresh': np.linspace(0.55, 0.75, 5), 'ret_thresh': np.linspace(0.001, 0.003, 5)}),
        # 6. EMA Cross + BE
        ("EMA Cross + BE", strat6_ema_cross_be,
         {'ema_ratio_thresh': np.linspace(1.01, 1.05, 5), 'price_vs_ema_thresh': np.linspace(1.005, 1.03, 5),
          'be_thresh': np.linspace(0.45, 0.55, 5)}),
        # 7. Multi‑timeframe Momentum
        ("Multi‑TF Mom", strat7_multi_timeframe,
         {'ret_5m_thresh': np.linspace(0.0005, 0.003, 6), 'ret_1m_thresh': np.linspace(0.0002, 0.0015, 5)}),
        # 8. Polymarket Cumulative Depth Imbalance
        ("Poly Depth Imb", strat8_poly_depth_imbalance,
         {'depth_imb_thresh': np.linspace(0.1, 0.4, 4)}),
        # 9. Volatility Adjusted Momentum
        ("Vol Adj Mom", strat9_volatility_adjusted,
         {'ret_5m_thresh_low': np.linspace(0.0005, 0.002, 4), 'vol_thresh_low': np.linspace(0.0002, 0.001, 4),
          'ret_5m_thresh_high': np.linspace(0.0015, 0.003, 4), 'vol_thresh_high': np.linspace(0.0005, 0.002, 4)}),
        # 10. Break‑even Only
        ("BE Only", strat10_break_even_only,
         {'be_thresh': np.linspace(0.40, 0.49, 5)}),
        # 11. Price vs EMA Reversion
        ("Price vs EMA Rev", strat11_price_vs_ema_reversion,
         {'price_dev_thresh': np.linspace(1.01, 1.05, 5), 'be_thresh': np.linspace(0.45, 0.55, 5)}),
        # 12. Trade Flow + OB Imbalance
        ("TradeFlow+OB Imb", strat12_trade_flow_ob_imb,
         {'trade_imb_thresh': np.linspace(0.1, 0.4, 4), 'ob_imb_thresh': np.linspace(0.1, 0.4, 4)}),
        # 13. Market Progress Momentum
        ("Market Progress Mom", strat13_market_progress_momentum,
         {'progress_thresh': np.linspace(0.3, 0.7, 5), 'ret_thresh': np.linspace(0.0005, 0.003, 6)}),
        # 14. Spread Filter
        ("Spread Filter", strat14_spread_filter,
         {'spread_thresh': np.linspace(0.0002, 0.001, 5), 'ret_thresh': np.linspace(0.0005, 0.003, 6)}),
        # 15. Volume Surge
        ("Volume Surge", strat15_volume_surge,
         {'volume_percentile': volume_percentiles, 'ret_thresh': np.linspace(0.0005, 0.003, 6)}),
    ]

    results = run_optimized_backtest(df, strategies)

    # Build summary table
    summary = pd.DataFrame({
        name: res['metrics'] for name, res in results.items()
    }).T
    summary = summary[['total_pnl', 'num_trades', 'win_rate', 'avg_pnl', 'sharpe']].round(4)
    print("\n===== Summary Table (Out‑of‑Sample) =====")
    print(summary)

    # Plot equity curves
    plot_equity_curves(results, PLOT_FILE)

if __name__ == '__main__':
    main()