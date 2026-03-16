- cd polymarket-bot
- git pull origin main
- source .venv/bin/activate
- rm bot_log.out
- Para lanzar en VPS y redirigir la salida: nohup python -u polymarket_collector_optimized.py > bot_log.log 2>&1 &
- Buscar el proceso: ps aux | grep bot.py
- Parar el proceso: kill XXXXXX
- Ver registro de logs: tail -n 100 nohup.out
- Salir de .venv: deactivate
- Traer fichero: scp root@116.203.230.26:~/polymarket-bot/polymarket_pro_dataset.csv .

# Dataset Variables

## Market Metadata

| Variable | Description |
|---|---|
| `timestamp` | UTC timestamp when the snapshot was collected. |
| `market_slug` | Unique identifier of the Polymarket BTC up/down market. |
| `seconds_left` | Seconds remaining until the market resolves. |

---

# BTC Price Features

| Variable | Description |
|---|---|
| `btc_spot` | Current BTC/USDT price from Binance. |
| `btc_price_market_open` | BTC price at the moment the Polymarket market opened. |
| `btc_return_since_open` | Percentage return of BTC since the market opened. |

---

# Polymarket Orderbook (UP Contract)

| Variable | Description |
|---|---|
| `up_bid_p_1` | Best bid price for the UP contract. |
| `up_bid_s_1` | Size available at the best bid level. |
| `up_bid_p_2` | Second best bid price. |
| `up_bid_s_2` | Size available at the second bid level. |
| `up_bid_p_3` | Third best bid price. |
| `up_bid_s_3` | Size available at the third bid level. |
| `up_ask_p_1` | Best ask price for the UP contract. |
| `up_ask_s_1` | Size available at the best ask level. |
| `up_ask_p_2` | Second best ask price. |
| `up_ask_s_2` | Size available at the second ask level. |
| `up_ask_p_3` | Third best ask price. |
| `up_ask_s_3` | Size available at the third ask level. |

---

# Polymarket Orderbook (DOWN Contract)

| Variable | Description |
|---|---|
| `down_bid_p_1` | Best bid price for the DOWN contract. |
| `down_bid_s_1` | Size available at the best bid level. |
| `down_bid_p_2` | Second best bid price. |
| `down_bid_s_2` | Size available at the second bid level. |
| `down_bid_p_3` | Third best bid price. |
| `down_bid_s_3` | Size available at the third bid level. |
| `down_ask_p_1` | Best ask price for the DOWN contract. |
| `down_ask_s_1` | Size available at the best ask level. |
| `down_ask_p_2` | Second best ask price. |
| `down_ask_s_2` | Size available at the second ask level. |
| `down_ask_p_3` | Third best ask price. |
| `down_ask_s_3` | Size available at the third ask level. |

---

# Polymarket Fees

| Variable | Description |
|---|---|
| `up_fee_bps` | Trading fee (in basis points) applied to the UP contract. |
| `down_fee_bps` | Trading fee (in basis points) applied to the DOWN contract. |

---

# Binance Orderbook Microstructure

| Variable | Description |
|---|---|
| `bid_volume_10` | Total bid-side liquidity across the top 10 levels of the Binance orderbook. |
| `ask_volume_10` | Total ask-side liquidity across the top 10 levels of the Binance orderbook. |
| `orderbook_imbalance` | Liquidity imbalance between bid and ask volumes. |
| `mid_price` | Midpoint price between the best bid and best ask. |
| `spread` | Difference between the best ask and best bid price. |

---

# Momentum Features

| Variable | Description |
|---|---|
| `ret_1m` | BTC price return over the last 1 minute. |
| `ret_3m` | BTC price return over the last 3 minutes. |
| `ret_5m` | BTC price return over the last 5 minutes. |
| `ret_10m` | BTC price return over the last 10 minutes. |

---

# Exponential Moving Averages

| Variable | Description |
|---|---|
| `ema_3` | 3-minute exponential moving average of BTC price. |
| `ema_6` | 6-minute exponential moving average of BTC price. |
| `ema_12` | 12-minute exponential moving average of BTC price. |
| `ema_24` | 24-minute exponential moving average of BTC price. |
| `ema_ratio` | Ratio between EMA(3) and EMA(12), used as a short-term momentum indicator. |
| `price_vs_ema12` | Ratio between current BTC price and the EMA(12). |

---

# Volume Features

| Variable | Description |
|---|---|
| `volume_1m` | Trading volume during the last 1 minute. |
| `volume_5m` | Total trading volume over the last 5 minutes. |
| `volume_10m` | Total trading volume over the last 10 minutes. |

---

# Volatility Features

| Variable | Description |
|---|---|
| `volatility_5m` | Standard deviation of BTC returns over the last 5 minutes. |
| `volatility_10m` | Standard deviation of BTC returns over the last 10 minutes. |

---

# Trade Flow Features

| Variable | Description |
|---|---|
| `buy_volume_last200` | Total buy-side trade volume from the last 200 Binance trades. |
| `sell_volume_last200` | Total sell-side trade volume from the last 200 Binance trades. |
| `trade_imbalance` | Imbalance between buy and sell trade volumes over the last 200 trades. |

Prompt:

Actúa como un Quant Developer especializado en HFT (High Frequency Trading) y Cripto.

Contexto del Dataset:
Tengo un dataset de Polymarket (mercados binarios UP/DOWN de Bitcoin) con snapshots cada 10 segundos. Contiene:

Timestamp y Market ID: Para identificar series temporales independientes.

BTC Spot: Precio de Binance en tiempo real.

Order Book L3: Precio y tamaño (size) de los 3 mejores niveles de Bid y Ask para ambos contratos (UP y DOWN).

Seconds Left: Tiempo restante para el vencimiento.

Resolution: Target (UP o DOWN).

Objetivo:
Crear un script en Python para un Notebook que entrene un XGBoost evitando estrictamente el Target Leakage.

Requerimientos Técnicos del Código:

Preprocesamiento: Ordenar por market_slug y timestamp. Asegurar que cualquier cálculo de ventana temporal se haga mediante .groupby('market_slug') para no mezclar datos de distintos mercados.

Feature Engineering (Microestructura):

L3 Imbalance: Calcular el desequilibrio de volumen sumando los 3 niveles de bid vs los 3 de ask.

Book Pressure: Diferencia de liquidez entre niveles (Nivel 1 vs Nivel 2 y 3).

Micro-price: Precio ajustado por el volumen de los bids/asks del nivel 1.

Spreads: Diferencia relativa entre bid y ask.

Feature Engineering (BTC Momentum):

EMAs Suavizadas: Calcular EMAs de 1 y 5 minutos del precio de BTC.

Log-Returns: Retorno logarítmico del BTC respecto a N periodos atrás (ej. 6 periodos = 1 min).

Distancia a la EMA: Ratio entre precio actual y la EMA.

Evitar Leakage en Inferencia: Todas las variables deben ser calculables con un rolling o ewm que simule la llegada de datos en streaming (usando solo el pasado).

Modelo:

Usar un Time Series Split (no aleatorio) para la validación.

Entrenar un XGBClassifier con early_stopping_rounds para evitar overfitting.

Incluir un gráfico de feature_importance basado en "Gain".

Salida: Dame el código limpio, optimizado para memoria y listo para ejecutarse.
