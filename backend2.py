import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import timedelta
from datetime import datetime

def get_tsx_tickers():
    url = "https://www.tsx.com/json/company-directory/search/tsx/^*"
    response = requests.get(url)
    data = response.json()
    tickers = [item['symbol'] for item in data['results'] if '.' not in item['symbol']]
    return tickers

def fetch_data(ticker_symbol):
    ticker_data = yf.Ticker(ticker_symbol)
    return ticker_data.history(period='1d', start='2020-1-1', end='2023-12-31')

def fetch_latest_data(ticker_symbol):
    today = datetime.today().strftime('%Y-%m-%d')
    ticker_data = yf.Ticker(ticker_symbol)
    return ticker_data.history(period='1d', start='2024-01-01', end=today)

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, span):
    return data['Close'].ewm(span=span, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_window, long_window, signal_window):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, window):
    middle_band = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = middle_band + (rolling_std * 2)
    lower_band = middle_band - (rolling_std * 2)
    return upper_band, lower_band
