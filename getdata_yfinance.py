
import polars as pl 
import pandas as pd 


import yfinance as yf

import warnings





start_date = '2000-01-01'
end_date = '2025-09-25'

startfile = "msds_getdata_yfinance_"




symbol = 'AAPL'
ticker = yf.Ticker(symbol)
historical_data = ticker.history(start = start_date, end = end_date)
print(historical_data.head())
historical_data.to_csv(startfile + symbol.lower() + ".csv")




