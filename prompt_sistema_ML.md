Actúa como un Senior Quant ML Engineer especializado en trading algorítmico, series temporales y backtesting.

Quiero que construyas un notebook de Jupyter en Python, completamente funcional y bien estructurado, para resolver este problema de predicción y simulación sobre mercados binarios de Polymarket BTC up/down de 5 minutos.

OBJETIVO
Tengo un dataset histórico con snapshots temporales (cada pocos segundos) del mercado y del subyacente BTC. Quiero un sistema de ML que haga 3 cosas:

1) Prediga cómo va a terminar el mercado al cierre (vairable 'resolution'):
   - clase binaria: UP / DOWN
   - probabilidad calibrada de UP y DOWN

2) Genere señales de entrada:
   - decidir cuándo entrar en UP o en DOWN
   - basándose en el ROI esperado neto
   - teniendo en cuenta fees, precio de entrada, liquidez y break-even
   - solo operar cuando el valor esperado sea positivo y supere un umbral configurable

3) Haga un backtest tipo simulación de equity:
   - simular entradas y salidas sobre el histórico
   - calcular PnL por operación
   - equity curve
   - drawdown
   - win rate
   - profit factor
   - ROI total
   - Sharpe simple si procede
   - número de trades
   - métricas por mercado y globales

REQUISITOS IMPORTANTES DEL MODELO
- Debe evitar leakage temporal.
- La validación debe hacerse respetando el orden temporal.
- Si hay varios snapshots por market_slug, no se debe mezclar información futura del mismo mercado en train y test.
- La partición debe hacerse por mercado y por tiempo, de forma robusta.
- El notebook debe incluir separación train / validation / test.
- Si usas calibración de probabilidades, hazla correctamente.
- Debe comparar al menos un baseline sencillo contra un modelo más potente.

MODELAJE
Quiero que propongas y pruebes, como mínimo:
- un baseline interpretable, por ejemplo Logistic Regression o XGBoost/LightGBM con features bien preparadas
- un modelo más potente, por ejemplo LightGBM, XGBoost o CatBoost
- si lo consideras útil, una versión con calibración de probabilidades

FEATURE ENGINEERING
Puedes crear features adicionales a partir de las variables existentes, por ejemplo:

TARGET / LABEL
Debes definir claramente el target principal:
- `y = 1` si el mercado termina UP
- `y = 0` si termina DOWN

Además, quiero que el notebook calcule para cada snapshot:
- probabilidad predicha de UP
- ROI esperado si entro en UP
- ROI esperado si entro en DOWN
- señal final de trading

LÓGICA DE LA SEÑAL
La decisión de entrada debe basarse en el valor esperado neto, no solo en la probabilidad.
Quiero que tengas en cuenta:
- `up_win_net`
- `up_loss_net`
- `down_win_net`
- `down_loss_net`
- `up_break_even`
- `down_break_even`

Ejemplo de criterio deseado:
- entrar en UP solo si el EV esperado de UP es positivo y mayor que un umbral mínimo
- entrar en DOWN solo si el EV esperado de DOWN es positivo y mayor que un umbral mínimo
- si ambos son positivos, elegir la mejor oportunidad por EV o ROI esperado
- si ninguno cumple el umbral, no operar

Quiero que la señal final pueda ser algo como:
- `LONG_UP`
- `LONG_DOWN`
- `NO_TRADE`

BACKTEST
El backtest debe simular una cartera con stake fijo o variable, configurable por parámetro.
Debe incluir:
- capital inicial
- stake por operación
- comisiones ya contempladas o añadidas si falta algo
- ejecución solo en la primera señal válida de cada mercado o una lógica coherente de entrada
- no permitir mirar el futuro
- salida de cada operación al cierre del mercado

Quiero que la simulación genere:
- tabla de trades
- curva de equity
- estadísticas resumidas
- análisis de sensibilidad de umbrales si es posible

MÉTRICAS
Debes reportar como mínimo:
- Accuracy
- ROC AUC
- PR AUC si aplica
- ROI medio por trade
- win rate
- max drawdown
- profit factor
- equity final

EXPLICABILIDAD
Incluye interpretabilidad del modelo:
- feature importance

ESTRUCTURA DEL NOTEBOOK
Quiero el notebook en secciones claras:
1. Imports y configuración
2. Carga de datos
3. Limpieza y validación de datos
4. Feature engineering
5. Definición del target
6. Split temporal / por mercado
7. Entrenamiento de modelos
8. Evaluación
9. Cálculo de probabilidades y EV
10. Generación de señales
11. Backtest de equity
12. Interpretabilidad
13. Conclusiones

ENTREGABLES
Necesito que me devuelvas:
- el notebook descargable
- explicaciones breves en Markdown para cada bloque
- código listo para ejecutarse en Jupyter
- sin pseudocódigo
- sin funciones incompletas
- sin omitir pasos críticos

RESTRICCIONES Y BUENAS PRÁCTICAS
- Usa pandas y scikit-learn como base.
- Usa random seeds.
- Maneja NaNs e infinitos.
- Normaliza o escala solo si el modelo lo requiere.
- No uses variables futuras para construir features.
- No hagas supuestos no justificados: explícitalos.

VARIABLES DEL DATASET
## Market Metadata

| Variable | Description |
|---|---|
| `timestamp` | UTC timestamp when the snapshot was collected. |
| `market_slug` | Unique identifier of the Polymarket BTC up/down 5-minute market. |
| `seconds_left` | Seconds remaining until the market resolves. |
| `market_progress` | Fraction of the market elapsed at snapshot time (0.0 at open → 1.0 at close). |
| `resolution` | up or down, describes how the market ends |

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