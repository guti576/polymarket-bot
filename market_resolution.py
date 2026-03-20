import pandas as pd
import requests
import json
import time

# -------- CONFIG --------
INPUT_FILE = "polymarket_dataset_5m.csv"
OUTPUT_FILE = "dataset_with_resolution.csv"
API_URL = "https://gamma-api.polymarket.com/markets?slug="


def get_market_resolution(slug):
    try:
        r = requests.get(API_URL + slug, timeout=10)
        data = r.json()

        if not data:
            return None

        market = data[0]

        outcomes = json.loads(market["outcomes"])
        prices = list(map(float, json.loads(market["outcomePrices"])))

        winner = outcomes[prices.index(max(prices))]
        return winner.lower()

    except Exception as e:
        print(f"Error with {slug}: {e}")
        return None


def main():

    # 1. Cargar dataset original
    df = pd.read_csv(INPUT_FILE)

    # 2. Obtener slugs únicos
    unique_slugs = df["market_slug"].dropna().unique()

    print(f"Unique markets to query: {len(unique_slugs)}")

    # 3. Construir dataset de resoluciones
    resolution_rows = []

    for slug in unique_slugs:
        print("Fetching:", slug)

        resolution = get_market_resolution(slug)

        resolution_rows.append({
            "market_slug": slug,
            "resolution": resolution
        })

        time.sleep(0.15)  # evitar rate limit

    resolution_df = pd.DataFrame(resolution_rows)

    # 4. Guardar dataset de resoluciones (opcional)
    resolution_df.to_csv("market_resolutions.csv", index=False)

    # 5. Join con dataset original
    df_final = df.merge(resolution_df, on="market_slug", how="left")

    # 6. Guardar resultado final
    df_final.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()