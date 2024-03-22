from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from keras.layers import Input

def fit_arima_model(data):
    model = ARIMA(data, order=(1, 1, 1))
    result = model.fit()
    return result

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1,1))
    return scaler, scaled_data

def create_lstm_dataset(scaled_data, time_step=1):
    X_data, y_data = [], []
    for i in range(len(scaled_data) - time_step - 1):
        a = scaled_data[i:(i+time_step), 0]
        X_data.append(a)
        y_data.append(scaled_data[i + time_step, 0])
    return np.array(X_data), np.array(y_data)

def create_lstm_model():
    model = Sequential()
    model.add(Input(shape=(100, 1)))  # Adjust the shape according to your data
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, X, y):
    model.fit(X, y, batch_size=1, epochs=1)
    return model

def predict_lstm(model, X, scaler):
    train_predict = model.predict(X)
    train_predict = scaler.inverse_transform(train_predict)
    return train_predict