# polymarket_binance_collector.py

import requests
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime

# =========================
# CONFIG
# =========================

DATASET_FILE = "./polymarket_dataset_full.csv"

SLEEP_SECONDS = 8
ORDERBOOK_DEPTH = 3

BINANCE_BASE = "https://api.binance.com"
BINANCE_PRICE = "https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT"

GAMMA_URL = "https://gamma-api.polymarket.com/events/slug"
BOOK_URL = "https://clob.polymarket.com/book"
FEE_RATE_URL = "https://clob.polymarket.com/fee-rate"

session = requests.Session()


# =========================
# BTC SPOT
# =========================

def get_btc_price():

    try:
        r = session.get(BINANCE_PRICE, timeout=5).json()
        return float(r["price"])
    except:
        return None


# =========================
# POLYMARKET MARKET INFO
# =========================

def get_current_market():

    now = int(time.time())

    epoch = now - (now % 900)

    slug = f"btc-updown-15m-{epoch}"

    try:

        data = session.get(f"{GAMMA_URL}/{slug}", timeout=5).json()

        market = data["markets"][0]

        token_ids = json.loads(market["clobTokenIds"])

        seconds_left = epoch + 900 - now

        return slug, token_ids[0], token_ids[1], seconds_left

    except:
        return None, None, None, None


# =========================
# POLYMARKET ORDERBOOK
# =========================

def get_orderbook_snapshot(token_id, prefix):

    row = {}

    try:

        r = session.get(f"{BOOK_URL}?token_id={token_id}", timeout=5).json()

        bids = r.get("bids", [])
        asks = r.get("asks", [])

        for i in range(ORDERBOOK_DEPTH):

            row[f"{prefix}_bid_p_{i+1}"] = float(bids[i]["price"]) if i < len(bids) else None
            row[f"{prefix}_bid_s_{i+1}"] = float(bids[i]["size"]) if i < len(bids) else None

            row[f"{prefix}_ask_p_{i+1}"] = float(asks[i]["price"]) if i < len(asks) else None
            row[f"{prefix}_ask_s_{i+1}"] = float(asks[i]["size"]) if i < len(asks) else None

    except:
        pass

    return row


# =========================
# FEES
# =========================

def get_fee_rate(token_id):

    try:
        r = session.get(f"{FEE_RATE_URL}?token_id={token_id}", timeout=5).json()
        return int(r.get("base_fee",0))
    except:
        return 0


# =========================
# BINANCE DEPTH FEATURES
# =========================

def get_binance_depth():

    try:

        r = session.get(f"{BINANCE_BASE}/api/v3/depth?symbol=BTCUSDT&limit=20", timeout=5).json()

        bids = np.array([[float(p),float(q)] for p,q in r["bids"]])
        asks = np.array([[float(p),float(q)] for p,q in r["asks"]])

        bid_vol = bids[:10,1].sum()
        ask_vol = asks[:10,1].sum()

        best_bid = bids[0,0]
        best_ask = asks[0,0]

        mid = (best_bid + best_ask)/2
        spread = best_ask - best_bid

        imbalance = (bid_vol - ask_vol)/(bid_vol + ask_vol)

        return {

            "bid_volume_10":bid_vol,
            "ask_volume_10":ask_vol,
            "orderbook_imbalance":imbalance,
            "mid_price":mid,
            "spread":spread

        }

    except:

        return {}


# =========================
# KLINES FEATURES
# =========================

def get_klines_features():

    try:

        r = session.get(
            f"{BINANCE_BASE}/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100",
            timeout=5
        ).json()

        df = pd.DataFrame(r)

        df = df[[0,1,2,3,4,5]]
        df.columns = ["time","open","high","low","close","volume"]

        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        price = df["close"].iloc[-1]

        returns = df["close"].pct_change()

        ret1 = returns.iloc[-1]
        ret3 = df["close"].pct_change(3).iloc[-1]
        ret5 = df["close"].pct_change(5).iloc[-1]
        ret10 = df["close"].pct_change(10).iloc[-1]

        ema3 = df["close"].ewm(span=3).mean().iloc[-1]
        ema6 = df["close"].ewm(span=6).mean().iloc[-1]
        ema12 = df["close"].ewm(span=12).mean().iloc[-1]
        ema24 = df["close"].ewm(span=24).mean().iloc[-1]

        vol1 = df["volume"].iloc[-1]
        vol5 = df["volume"].tail(5).sum()
        vol10 = df["volume"].tail(10).sum()

        vol_std5 = returns.tail(5).std()
        vol_std10 = returns.tail(10).std()

        return {

            "ret_1m":ret1,
            "ret_3m":ret3,
            "ret_5m":ret5,
            "ret_10m":ret10,

            "ema_3":ema3,
            "ema_6":ema6,
            "ema_12":ema12,
            "ema_24":ema24,

            "ema_ratio":ema3/ema12,
            "price_vs_ema12":price/ema12,

            "volume_1m":vol1,
            "volume_5m":vol5,
            "volume_10m":vol10,

            "volatility_5m":vol_std5,
            "volatility_10m":vol_std10
        }

    except:

        return {}


# =========================
# TRADE FLOW
# =========================

def get_trade_flow():

    try:

        r = session.get(
            f"{BINANCE_BASE}/api/v3/trades?symbol=BTCUSDT&limit=200",
            timeout=5
        ).json()

        buy = 0
        sell = 0

        for t in r:

            qty = float(t["qty"])

            if t["isBuyerMaker"]:
                sell += qty
            else:
                buy += qty

        total = buy + sell

        imbalance = (buy - sell)/total if total > 0 else 0

        return {

            "buy_volume_last200":buy,
            "sell_volume_last200":sell,
            "trade_imbalance":imbalance

        }

    except:

        return {}


# =========================
# MAIN LOOP
# =========================

def main():

    print("collector started")

    file_exists = os.path.isfile(DATASET_FILE)

    current_slug = None

    market_open_price = None

    while True:

        try:

            slug, tk_up, tk_down, secs_left = get_current_market()

            if slug is None:
                time.sleep(SLEEP_SECONDS)
                continue

            btc_price = get_btc_price()

            if slug != current_slug:

                print("new market", slug)

                current_slug = slug

                market_open_price = btc_price

            row = {

                "timestamp":datetime.utcnow().isoformat(),

                "market_slug":slug,

                "btc_spot":btc_price,

                "btc_price_market_open":market_open_price,

                "btc_return_since_open":(
                    btc_price/market_open_price - 1
                    if market_open_price else None
                ),

                "seconds_left":secs_left

            }

            row.update(get_orderbook_snapshot(tk_up,"up"))
            row.update(get_orderbook_snapshot(tk_down,"down"))

            row["up_fee_bps"] = get_fee_rate(tk_up)
            row["down_fee_bps"] = get_fee_rate(tk_down)

            row.update(get_binance_depth())

            row.update(get_klines_features())

            row.update(get_trade_flow())

            df = pd.DataFrame([row])

            df.to_csv(

                DATASET_FILE,
                mode="a",
                header=not file_exists,
                index=False

            )

            file_exists = True

            print("snapshot saved", row["timestamp"], end="\r")

        except Exception as e:

            print("error",e)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()