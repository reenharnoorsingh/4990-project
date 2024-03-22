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

# #ARIMA MODEL
# #import ARIMA model
# from statsmodels.tsa.arima.model import ARIMA
# # Fit the ARIMA model for ticker 1
# arima_model1 = ARIMA(ticker_df1['Close'], order=(1, 1, 1))
# arima_result1 = arima_model1.fit()

# # Fit the ARIMA model for ticker 2
# arima_model2 = ARIMA(ticker_df2['Close'], order=(1, 1, 1))
# arima_result2 = arima_model2.fit()
# plt.figure(figsize=(14, 7))

# # Plot the original closing prices and ARIMA fitted values for ticker 1
# plt.subplot(2, 1, 1)
# plt.plot(ticker_df1.index, ticker_df1['Close'], label='Original')
# plt.plot(ticker_df1.index, arima_result1.fittedvalues, label='ARIMA Fitted', color='red')
# plt.title(f'ARIMA Model - {tickerSymbol1}')
# plt.xlabel('Date')
# plt.ylabel('Closing Price')
# plt.legend()

# # Plot the original closing prices and ARIMA fitted values for ticker 2
# plt.subplot(2, 1, 2)
# plt.plot(ticker_df2.index, ticker_df2['Close'], label='Original')
# plt.plot(ticker_df2.index, arima_result2.fittedvalues, label='ARIMA Fitted', color='red')
# plt.title(f'ARIMA Model - {tickerSymbol2}')
# plt.xlabel('Date')
# plt.ylabel('Closing Price')
# plt.legend()

# plt.tight_layout()
# plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data1 = scaler.fit_transform(ticker_df1['Close'].values.reshape(-1,1))
scaled_data2 = scaler.fit_transform(ticker_df2['Close'].values.reshape(-1,1))

# Define a function to create a dataset for LSTM
def create_dataset(scaled_data, time_step=1):
    X_data, y_data = [], []
    for i in range(len(scaled_data) - time_step - 1):
        a = scaled_data[i:(i+time_step), 0]
        X_data.append(a)
        y_data.append(scaled_data[i + time_step, 0])
    return np.array(X_data), np.array(y_data)

time_step = 100
X1, y1 = create_dataset(scaled_data1, time_step)
X2, y2 = create_dataset(scaled_data2, time_step)

# Reshape into [samples, time steps, features] required for LSTM
X1 = X1.reshape(X1.shape[0],X1.shape[1] , 1)
X2 = X2.reshape(X2.shape[0],X2.shape[1] , 1)

from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define a function to create an LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model1 = create_lstm_model()
model2 = create_lstm_model()

# Train the models (for simplicity, not splitting into train and test)
model1.fit(X1, y1, batch_size=1, epochs=1)
model2.fit(X2, y2, batch_size=1, epochs=1)

# Making predictions
train_predict1 = model1.predict(X1)
train_predict1 = scaler.inverse_transform(train_predict1)  # Inverse transform to get actual value

train_predict2 = model2.predict(X2)
train_predict2 = scaler.inverse_transform(train_predict2)  # Inverse transform to get actual value

# Plot the results
plt.figure(figsize=(16,8))

# Ticker 1
plt.subplot(2, 1, 1)
plt.plot(ticker_df1['Close'], label='Actual Price')
plt.plot(ticker_df1.index[time_step+1:len(train_predict1)+time_step+1], train_predict1, label='Predicted Price')
plt.title(f'{tickerSymbol1} Closing Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Ticker 2
plt.subplot(2, 1, 2)
plt.plot(ticker_df2['Close'], label='Actual Price')
plt.plot(ticker_df2.index[time_step+1:len(train_predict2)+time_step+1], train_predict2, label='Predicted Price')
plt.title(f'{tickerSymbol2} Closing Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.show()
