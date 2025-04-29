import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def evaluate_arima(model_fit, test_series):
    predictions = model_fit.forecast(steps=len(test_series))
    mse = mean_squared_error(test_series, predictions)
    mae = mean_absolute_error(test_series, predictions)
    return predictions, mse, mae


def evaluate_lstm(model, X_test, y_test, scaler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_tensor).cpu().numpy()

    # Rescale predictions and ground truth if scaler is provided
    if scaler:
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return predictions, mse, mae


def plot_predictions(actual, predicted, title="Model Predictions"):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
