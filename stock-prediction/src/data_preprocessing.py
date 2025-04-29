import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

def load_clean_data(filepath):
      # Load the dataset
      df = pd.read_csv(filepath, skiprows=2, parse_dates=['Date'], index_col='Date')
      # Rename columns
      df.columns = ["close", "high", "low", "open", "volume"]
      return df

def get_arima_ready_series(df: pd.DataFrame) -> pd.Series:
    series = df["Close"].dropna()
    diff_series = series.diff().dropna()  # First-order differencing
    return diff_series

def get_lstm_ready_data(df, window_size=60):
      """ Prepare data for LSTM model """
      series = df["close"].values.reshape(-1, 1)
      scaler = MinMaxScaler()
      scaled_series = scaler.fit_transform(series)
      
      X, y = [], []
      for i in range(window_size, len(scaled_series)):
            X.append(scaled_series[i - window_size : i])
            y.append(scaled_series[i])

      X, y = np.array(X), np.array(y)
      return X, y, scaler

def split_data(X, y, train_ratio: float = 0.8):
    split_index = int(len(X) * train_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test
