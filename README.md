cd polymarket-bot
git pull origin main
source .venv/bin/activate
rm bot_log.out
Para lanzar en VPS y redirigir la salida: nohup python -u polymarket_collector_optimized.py > bot_log.log 2>&1 &
Buscar el proceso: ps aux | grep .py
Parar el proceso: kill XXXXXX
Ver registro de logs: tail -n 100 nohup.out
Salir de .venv: deactivate
Traer fichero: scp root@116.203.230.26:~/polymarket-bot/polymarket_pro_dataset.csv .

Run signal: python signal_engine.py --model-dir ./model --poll 1.0

# Estrategia en Producción: Strong Momentum + Mom Reversal Exit

**Fichero:** `signal_engine.py`
**Entrada:** `polymarket_dataset_5m.csv` (snapshots en tiempo real)
**Salida:** `trades_5m.csv` (señales accionables)

## Entrada (Strong Momentum)

Monitoriza mercados BTC UP/DOWN de 5 minutos en Polymarket. Cuando `btc_return_since_open` supera el percentil 75 histórico → `LONG_UP`. Cuando cae por debajo del percentil 25 → `LONG_DOWN`. Los umbrales se auto-calibran con una ventana rolling de 500 snapshots. Solo opera entre el 10% y 60% de vida del mercado. Un trade por mercado.

## Salida (Momentum Reversal)

Si el retorno de BTC cruza cero en contra con un margen de 0.0001 y el mercado ha superado el 15% de progreso → emite `EXIT_UP` o `EXIT_DOWN` para vender shares al bid y cortar pérdidas.

## Restricciones

- No genera trades con datos anteriores al lanzamiento del script (el histórico solo alimenta el calibrador)
- Stake: 10 USDC por trade, capital: 100€
- El CSV solo contiene filas accionables (`LONG_*` / `EXIT_*`)
- Dependencias: `pandas`, `numpy`

# Dataset Variables

## Market Metadata

| Variable | Description |
|---|---|
| `timestamp` | UTC timestamp when the snapshot was collected. |
| `market_slug` | Unique identifier of the Polymarket BTC up/down 5-minute market. |
| `seconds_left` | Seconds remaining until the market resolves. |
| `market_progress` | Fraction of the market elapsed at snapshot time (0.0 at open → 1.0 at close). |

---

## BTC Price Features

| Variable | Description |
|---|---|
| `btc_spot` | Current BTC/USDT price from Binance at snapshot time. |
| `btc_price_market_open` | BTC/USDT price at the moment the Polymarket market opened. |
| `btc_return_since_open` | Percentage return of BTC from market open to snapshot time. Positive means BTC is currently above the opening price (favours UP). |

---

## Polymarket Order Book — UP Contract

| Variable | Description |
|---|---|
| `up_bid_p_1` | Best bid price for the UP contract (highest price a buyer is willing to pay). |
| `up_bid_s_1` | Size (USDC) available at the best bid level. |
| `up_bid_p_2` | Second best bid price. |
| `up_bid_s_2` | Size available at the second bid level. |
| `up_bid_p_3` | Third best bid price. |
| `up_bid_s_3` | Size available at the third bid level. |
| `up_ask_p_1` | Best ask price for the UP contract (lowest price a seller is willing to accept). Equals the entry price if buying UP at market. |
| `up_ask_s_1` | Size (USDC) available at the best ask level. |
| `up_ask_p_2` | Second best ask price. |
| `up_ask_s_2` | Size available at the second ask level. |
| `up_ask_p_3` | Third best ask price. |
| `up_ask_s_3` | Size available at the third ask level. |

---

## Polymarket Order Book — DOWN Contract

| Variable | Description |
|---|---|
| `down_bid_p_1` | Best bid price for the DOWN contract. |
| `down_bid_s_1` | Size (USDC) available at the best bid level. |
| `down_bid_p_2` | Second best bid price. |
| `down_bid_s_2` | Size available at the second bid level. |
| `down_bid_p_3` | Third best bid price. |
| `down_bid_s_3` | Size available at the third bid level. |
| `down_ask_p_1` | Best ask price for the DOWN contract. Equals the entry price if buying DOWN at market. |
| `down_ask_s_1` | Size (USDC) available at the best ask level. |
| `down_ask_p_2` | Second best ask price. |
| `down_ask_s_2` | Size available at the second ask level. |
| `down_ask_p_3` | Third best ask price. |
| `down_ask_s_3` | Size available at the third ask level. |

---

## Polymarket Fees

| Variable | Description |
|---|---|
| `up_fee_bps` | Trading fee in basis points applied to the UP contract (1 bps = 0.01%). Cached once per market at open. |
| `down_fee_bps` | Trading fee in basis points applied to the DOWN contract. |

---

## Trade Opportunity — UP Contract

Computed from the UP order book at each snapshot. Simulates a market order of `STAKE` USDC consuming ask levels sequentially, exactly as Polymarket executes it in the UI. No additional HTTP calls.

| Variable | Description |
|---|---|
| `up_avg_fill` | Average execution price of a market order of `STAKE` USDC on the UP contract, weighted across consumed ask levels. May differ from `up_ask_p_1` if the stake exceeds the liquidity at the best ask. |
| `up_win_net` | Net profit (USDC) if the market resolves UP, after fees. Calculated as: `total_shares_received − stake − fee`. |
| `up_loss_net` | Net loss (USDC) if the market resolves DOWN, after fees. Always negative. Calculated as: `−stake − fee`. |
| `up_break_even` | Minimum win rate required for the UP bet to have zero expected value: `abs(up_loss_net) / (up_win_net + abs(up_loss_net))`. A model must predict UP with accuracy above this threshold to justify the trade. |

---

## Trade Opportunity — DOWN Contract

Same logic as the UP block, applied to the DOWN contract.

| Variable | Description |
|---|---|
| `down_avg_fill` | Average execution price of a market order of `STAKE` USDC on the DOWN contract. |
| `down_win_net` | Net profit (USDC) if the market resolves DOWN, after fees. |
| `down_loss_net` | Net loss (USDC) if the market resolves UP, after fees. Always negative. |
| `down_break_even` | Minimum win rate required for the DOWN bet to have zero expected value. |

---

## Binance Order Book Microstructure

Snapshot of the Binance BTC/USDT spot order book at the time of collection.

| Variable | Description |
|---|---|
| `bid_volume_10` | Total bid-side liquidity (BTC) across the top 10 levels of the Binance order book. |
| `ask_volume_10` | Total ask-side liquidity (BTC) across the top 10 levels of the Binance order book. |
| `orderbook_imbalance` | Liquidity imbalance: `(bid_volume_10 − ask_volume_10) / (bid_volume_10 + ask_volume_10)`. Positive means more buy-side pressure. |
| `mid_price` | Midpoint between the Binance best bid and best ask. |
| `spread` | Difference between the Binance best ask and best bid. A widening spread indicates lower liquidity or higher uncertainty. |

---

## Momentum Features

Computed from Binance 1-minute klines. All values use the last **closed** candle (`iloc[-2]`) to avoid noise from the incomplete current candle.

| Variable | Description |
|---|---|
| `ret_1m` | BTC price return over the last 1 closed minute (~20% of the 5m market duration). |
| `ret_3m` | BTC price return over the last 3 closed minutes (~60% of the market duration). |
| `ret_5m` | BTC price return over the last 5 closed minutes (covers the full market duration). |
| `ret_10m` | BTC price return over the last 10 closed minutes (pre-market trend context). |

---

## Exponential Moving Averages

| Variable | Description |
|---|---|
| `ema_3` | 3-period EMA of BTC close price (last closed candle). |
| `ema_6` | 6-period EMA of BTC close price. |
| `ema_12` | 12-period EMA of BTC close price. |
| `ema_24` | 24-period EMA of BTC close price. |
| `ema_ratio` | Ratio of EMA(3) to EMA(12). Values above 1 indicate short-term upward momentum. |
| `price_vs_ema12` | Ratio of current BTC price to EMA(12). Values above 1 indicate price is above its medium-term average. |

---

## Volume Features

| Variable | Description |
|---|---|
| `volume_1m` | BTC trading volume during the last closed 1-minute candle. |
| `volume_3m` | Total BTC trading volume over the last 3 closed minutes. |
| `volume_5m` | Total BTC trading volume over the last 5 closed minutes. |

---

## Volatility Features

| Variable | Description |
|---|---|
| `volatility_3m` | Standard deviation of BTC 1-minute returns over the last 3 closed candles. |
| `volatility_5m` | Standard deviation of BTC 1-minute returns over the last 5 closed candles. |

---

## Trade Flow Features

Derived from the last 200 individual trades on Binance. `isBuyerMaker = true` means the buyer was the passive (maker) side, so the aggressive initiator was a seller.

| Variable | Description |
|---|---|
| `buy_volume_last200` | Total BTC volume of aggressively initiated buy trades in the last 200 trades. |
| `sell_volume_last200` | Total BTC volume of aggressively initiated sell trades in the last 200 trades. |
| `trade_imbalance` | Net directional flow: `(buy − sell) / (buy + sell)`. Positive means buyers are more aggressive. |
| `ret_30s` | BTC price return over the last 30 seconds, computed from individual trade timestamps. Captures sub-minute momentum not visible in 1m klines. Represents ~10% of the 5-minute market window. |