import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def get_tsx_tickers():
    url = "https://www.tsx.com/json/company-directory/search/tsx/^*"
    response = requests.get(url)
    data = response.json()
    tickers = [item['symbol'] for item in data['results'] if '.' not in item['symbol']]
    return tickers

def fetch_data(ticker_symbol):
    ticker_data = yf.Ticker(ticker_symbol)
    return ticker_data.history(period='1d', start='2020-1-1', end='2023-12-31')

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

def create_lstm_dataset(data, time_step=100):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X_data, y_data = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X_data.append(scaled_data[i:(i + time_step), 0])
        y_data.append(scaled_data[i + time_step, 0])
    
    X_data, y_data = np.array(X_data), np.array(y_data)
    X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
    return X_data, y_data, scaler

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, X_train, y_train, epochs=1, batch_size=1):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict_lstm(model, X_data, scaler):
    predictions = model.predict(X_data)
    return scaler.inverse_transform(predictions)
