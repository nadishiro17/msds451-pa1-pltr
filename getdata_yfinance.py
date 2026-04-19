#!/usr/bin/env python
# coding: utf-8

# # <span style="color:#4E2A84;"> MSDS Financial Engineering: Get Data from Yahoo! Finance

# ## Overview

# Financial time series data can be retrieved using the Python package yfinance, which interacts with the Yahoo! Finance application programming interface (API). These are public-domain data showing historical prices for individual markets or companies, identified by ticker codes.
# 
# Here is the Yahoo! Finance [Ticker Lookup](https://finance.yahoo.com/lookup/).

# ## Instructions

# To complete the assignment, follow these steps:

# ### Import
# 
# Run the next code cell to bring in the necessary Python packages.

# In[1]:


# import Python packages
import polars as pl # pandas DataFrame utilities
import pandas as pd # pandas DataFrame utilities

# package for downloading historical market data
import yfinance as yf

import warnings
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ### Settings for All Data Gathering
# Run the next code cell to define settings that apply to all the companies or markets for which we are gathering data.

# In[2]:


start_date = '2000-01-01'
end_date = '2025-09-25'

startfile = "msds_getdata_yfinance_"


# ### Read data for Apple 
# 
# The following code cell gathers pricing data for Apple and stores those data in a comma-delimited text file in the working directory app. 
# 
# Subsequent code cells do the same thing for other companies.
# 
# We check the contents of the working directory app after we get the data for each company.

# In[3]:


symbol = 'AAPL'
ticker = yf.Ticker(symbol)
historical_data = ticker.history(start = start_date, end = end_date)
print(historical_data.head())
historical_data.to_csv(startfile + symbol.lower() + ".csv")


# In[4]:


# check that the file is in the working directory
get_ipython().system('ls')

