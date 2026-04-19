import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

SYMBOL = "PLTR"
START_DATE = "2020-09-30"
END_DATE = "2026-04-19"

OUT_FILE = f"msds_getdata_yfinance_{SYMBOL.lower()}.csv"


def main() -> None:
    ticker = yf.Ticker(SYMBOL)
    historical_data = ticker.history(start=START_DATE, end=END_DATE)
    print(f"Downloaded {len(historical_data)} rows for {SYMBOL}")
    print(historical_data.head())
    print(historical_data.tail())
    historical_data.to_csv(OUT_FILE)
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
