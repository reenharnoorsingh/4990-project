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
plt.title('Stock Price Comparison {} and {} - Time Period 2020-2023'.format(tickerSymbol1, tickerSymbol2))
plt.legend()
plt.show()

# Calculate the 50-day Simple Moving Average (SMA) and Exponential Moving Average (EMA)
ticker_df1['50-day SMA'] = ticker_df1['Close'].rolling(window=50).mean()
ticker_df2['50-day SMA'] = ticker_df2['Close'].rolling(window=50).mean()

ticker_df1['50-day EMA'] = ticker_df1['Close'].ewm(span=50, adjust=False).mean()
ticker_df2['50-day EMA'] = ticker_df2['Close'].ewm(span=50, adjust=False).mean()

# Calculate the 200-day Simple Moving Average (SMA)
ticker_df1['200-day SMA'] = ticker_df1['Close'].rolling(window=200).mean()
ticker_df2['200-day SMA'] = ticker_df2['Close'].rolling(window=200).mean()

# Plot the 50-day SMA and EMA
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(ticker_df1.index, ticker_df1['Close'], label='{} Close Price'.format(tickerSymbol1))
plt.plot(ticker_df2.index, ticker_df2['Close'], label='{} Close Price'.format(tickerSymbol2))
plt.plot(ticker_df1.index, ticker_df1['50-day SMA'], linestyle='--', label='{} 50-day SMA'.format(tickerSymbol1))
plt.plot(ticker_df2.index, ticker_df2['50-day SMA'], linestyle='--', label='{} 50-day SMA'.format(tickerSymbol2))
plt.plot(ticker_df1.index, ticker_df1['50-day EMA'], linestyle='--', label='{} 50-day EMA'.format(tickerSymbol1))
plt.plot(ticker_df2.index, ticker_df2['50-day EMA'], linestyle='--', label='{} 50-day EMA'.format(tickerSymbol2))
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Comparison with 50-day SMA and EMA - {} and {}'.format(tickerSymbol1, tickerSymbol2))
plt.legend()

# Plot the 200-day SMA
plt.subplot(2, 1, 2)
plt.plot(ticker_df1.index, ticker_df1['Close'], label='{} Close Price'.format(tickerSymbol1))
plt.plot(ticker_df2.index, ticker_df2['Close'], label='{} Close Price'.format(tickerSymbol2))
plt.plot(ticker_df1.index, ticker_df1['200-day SMA'], linestyle='--', label='{} 200-day SMA'.format(tickerSymbol1))
plt.plot(ticker_df2.index, ticker_df2['200-day SMA'], linestyle='--', label='{} 200-day SMA'.format(tickerSymbol2))
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Comparison with 200-day SMA - {} and {}'.format(tickerSymbol1, tickerSymbol2))
plt.legend()

def calculate_rsi(data, window=14):
    close_price = data['Close']
    delta = close_price.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate RSI for both tickers
ticker_df1['RSI'] = calculate_rsi(ticker_df1)
ticker_df2['RSI'] = calculate_rsi(ticker_df2)

# Plot the RSI
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(ticker_df1.index, ticker_df1['RSI'], label='{} RSI'.format(tickerSymbol1))
plt.plot(ticker_df2.index, ticker_df2['RSI'], label='{} RSI'.format(tickerSymbol2))
plt.axhline(70, color='r', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='g', linestyle='--', label='Oversold (30)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.title('Relative Strength Index (RSI) - {} and {}'.format(tickerSymbol1, tickerSymbol2))
plt.legend()

short_window = 12
long_window = 26
signal_window = 9

ticker_df1['Short_MA'] = ticker_df1['Close'].ewm(span=short_window, adjust=False).mean()
ticker_df1['Long_MA'] = ticker_df1['Close'].ewm(span=long_window, adjust=False).mean()
ticker_df1['MACD'] = ticker_df1['Short_MA'] - ticker_df1['Long_MA']
ticker_df1['Signal_Line'] = ticker_df1['MACD'].ewm(span=signal_window, adjust=False).mean()

ticker_df2['Short_MA'] = ticker_df2['Close'].ewm(span=short_window, adjust=False).mean()
ticker_df2['Long_MA'] = ticker_df2['Close'].ewm(span=long_window, adjust=False).mean()
ticker_df2['MACD'] = ticker_df2['Short_MA'] - ticker_df2['Long_MA']
ticker_df2['Signal_Line'] = ticker_df2['MACD'].ewm(span=signal_window, adjust=False).mean()

# Plot the MACD
plt.subplot(3, 1, 3)
plt.plot(ticker_df1.index, ticker_df1['MACD'], label='{} MACD'.format(tickerSymbol1))
plt.plot(ticker_df1.index, ticker_df1['Signal_Line'], label='{} Signal Line'.format(tickerSymbol1))
plt.plot(ticker_df2.index, ticker_df2['MACD'], label='{} MACD'.format(tickerSymbol2))
plt.plot(ticker_df2.index, ticker_df2['Signal_Line'], label='{} Signal Line'.format(tickerSymbol2))
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Date')
plt.ylabel('MACD')
plt.title('Moving Average Convergence Divergence (MACD) - {} and {}'.format(tickerSymbol1, tickerSymbol2))
plt.legend()

plt.tight_layout()
plt.show()


