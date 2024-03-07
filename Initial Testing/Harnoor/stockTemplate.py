#imports
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import os

# Read the CSV file into a DataFrame
df = pd.read_csv('tsx_tickers.csv')

# Fetch a specific row by index
row_index1 = 0  # Replace with the desired row index
row_index2 = 1  # Replace with the desired row index

tickerSymbol1 = df.loc[row_index1, 'Ticker']
print(tickerSymbol1)
tickerSymbol2 = df.loc[row_index2, 'Ticker']
print(tickerSymbol2)

# Fetch the data for the first ticker
ticker_data1 = yf.Ticker(tickerSymbol1)
ticker_df1 = ticker_data1.history(period='1d', start='2020-1-1', end='2023-12-31')

# Fetch the data for the second ticker
ticker_data2 = yf.Ticker(tickerSymbol2)
ticker_df2 = ticker_data2.history(period='1d', start='2020-1-1', end='2023-12-31')

print(ticker_df1.head())

# Plot the closing prices
plt.plot(ticker_df1.index, ticker_df1['Close'], label=tickerSymbol1)
plt.plot(ticker_df2.index, ticker_df2['Close'], label=tickerSymbol2)
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title(f'Stock Price Comparison {tickerSymbol1} and {tickerSymbol2} - Time Period 2020-2023')
plt.legend()
plt.show()


#BOLLINGER BANDS ANALYSIS
# Calculate the rolling mean and standard deviation
# Calculate the rolling mean and standard deviation for ticker 1
ticker_df1['Rolling Mean'] = ticker_df1['Close'].rolling(window=20).mean()
ticker_df1['Rolling Std'] = ticker_df1['Close'].rolling(window=20).std()

# Calculate the upper and lower Bollinger Bands for ticker 1
ticker_df1['Upper Band'] = ticker_df1['Rolling Mean'] + (2 * ticker_df1['Rolling Std'])
ticker_df1['Lower Band'] = ticker_df1['Rolling Mean'] - (2 * ticker_df1['Rolling Std'])

# Calculate the rolling mean and standard deviation for ticker 2
ticker_df2['Rolling Mean'] = ticker_df2['Close'].rolling(window=20).mean()
ticker_df2['Rolling Std'] = ticker_df2['Close'].rolling(window=20).std()

# Calculate the upper and lower Bollinger Bands for ticker 2
ticker_df2['Upper Band'] = ticker_df2['Rolling Mean'] + (2 * ticker_df2['Rolling Std'])
ticker_df2['Lower Band'] = ticker_df2['Rolling Mean'] - (2 * ticker_df2['Rolling Std'])


# Open both Bollinger Bands graphs at the same time
plt.figure(figsize=(10, 6))

# Plot the Bollinger Bands for ticker 1
plt.subplot(2, 1, 1)
plt.plot(ticker_df1.index, ticker_df1['Close'], label=tickerSymbol1)
plt.plot(ticker_df1.index, ticker_df1['Upper Band'], label='Upper Band')
plt.plot(ticker_df1.index, ticker_df1['Lower Band'], label='Lower Band')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title(f'Bollinger Bands Analysis - {tickerSymbol1}')
plt.legend()

# Plot the Bollinger Bands for ticker 2
plt.subplot(2, 1, 2)
plt.plot(ticker_df2.index, ticker_df2['Close'], label=tickerSymbol2)
plt.plot(ticker_df2.index, ticker_df2['Upper Band'], label='Upper Band')
plt.plot(ticker_df2.index, ticker_df2['Lower Band'], label='Lower Band')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title(f'Bollinger Bands Analysis - {tickerSymbol2}')
plt.legend()

plt.tight_layout()
plt.show()
