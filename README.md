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
