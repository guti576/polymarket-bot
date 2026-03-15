
"""
Polymarket BTC Up/Down 15m Orderbook Collector (Optimized)

Key improvements:
- Persistent HTTP session (connection reuse)
- Centralized request helper with timeout + error handling
- Reduced repeated file checks
- Configurable parameters
- Clear structure and logging
"""

import requests
import pandas as pd
import time
import json
import os
from datetime import datetime

# =========================
# Configuration
# =========================
DATASET_FILE = "polymarket_pro_dataset.csv"
BINANCE_URL = "https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT"
GAMMA_URL = "https://gamma-api.polymarket.com/events/slug"
BOOK_URL = "https://clob.polymarket.com/book"

SLEEP_SECONDS = 10
ORDERBOOK_DEPTH = 3
REQUEST_TIMEOUT = 5

# =========================
# HTTP Session (faster)
# =========================
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


def safe_get(url):
    """Wrapper around requests.get with error handling."""
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


# =========================
# Data Fetching Functions
# =========================
def get_btc_spot_price():
    data = safe_get(BINANCE_URL)
    if data:
        try:
            return float(data["price"])
        except (KeyError, ValueError):
            pass
    return None


def get_current_market():
    now = int(time.time())
    epoch = now - (now % 900)  # 15 minute bucket
    slug = f"btc-updown-15m-{epoch}"

    data = safe_get(f"{GAMMA_URL}/{slug}")
    if not data:
        return None, None, None, None

    try:
        market = data["markets"][0]
        token_ids = json.loads(market["clobTokenIds"])
        seconds_left = epoch + 900 - now

        return slug, token_ids[0], token_ids[1], seconds_left
    except (KeyError, IndexError, json.JSONDecodeError):
        return None, None, None, None


def get_orderbook_snapshot(token_id, prefix):
    data = safe_get(f"{BOOK_URL}?token_id={token_id}")
    snapshot = {}

    if not data:
        return snapshot

    bids = data.get("bids", [])
    asks = data.get("asks", [])

    for i in range(ORDERBOOK_DEPTH):
        snapshot[f"{prefix}_bid_p_{i+1}"] = float(bids[i]["price"]) if i < len(bids) else None
        snapshot[f"{prefix}_bid_s_{i+1}"] = float(bids[i]["size"]) if i < len(bids) else None
        snapshot[f"{prefix}_ask_p_{i+1}"] = float(asks[i]["price"]) if i < len(asks) else None
        snapshot[f"{prefix}_ask_s_{i+1}"] = float(asks[i]["size"]) if i < len(asks) else None

    return snapshot


# =========================
# Main Collector Loop
# =========================
def main():
    print(f"🚀 Collector started. File: {DATASET_FILE}")

    current_slug = None
    file_exists = os.path.isfile(DATASET_FILE)

    while True:
        try:
            slug, tk_up, tk_down, secs_left = get_current_market()

            if slug:
                if slug != current_slug:
                    print(f"\n[NEW MARKET] {slug}")
                    current_slug = slug

                btc_price = get_btc_spot_price()

                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "market_slug": slug,
                    "btc_spot": btc_price,
                    "seconds_left": secs_left
                }

                row.update(get_orderbook_snapshot(tk_up, "up"))
                row.update(get_orderbook_snapshot(tk_down, "down"))

                df = pd.DataFrame([row])

                df.to_csv(
                    DATASET_FILE,
                    mode="a",
                    index=False,
                    header=not file_exists
                )

                file_exists = True

                print(
                    f"Snapshot saved: {row['timestamp']} | BTC: ${btc_price}",
                    end="\r"
                )

        except Exception as e:
            print(f"\n❌ Error: {e}")

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
