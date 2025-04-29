import pandas as pd
from src.data_preprocessing import load_clean_data, get_arima_ready_series, get_lstm_ready_data, split_data
from src.model_selection import get_lstm_hyperparams
from src.train_model import LSTMModel, train_lstm, save_lstm_model
from src.evaluate_model import evaluate_lstm, plot_predictions

"""
def run_arima_pipeline():
    df = load_clean_data("../data/raw/GOOG.csv")
    series = get_arima_ready_series(df)
    order = find_best_arima_params(series)
    model_fit = train_arima(series, order)
    save_arima_model(model_fit)

    test_size = int(len(series) * 0.2)
    test_series = series[-test_size:]
    predictions, mse, mae = evaluate_arima(model_fit, test_series)
    print(f"ARIMA MSE: {mse:.4f}, MAE: {mae:.4f}")
    plot_predictions(test_series, predictions, title="ARIMA Predictions")
"""

def run_lstm_pipeline():
    df = load_clean_data("data/raw/GOOG.csv")
    params = get_lstm_hyperparams()
    X, y, scaler = get_lstm_ready_data(df, params["window_size"])
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = LSTMModel(
        input_size=1,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"]
    )

    trained_model = train_lstm(
        model,
        X_train,
        y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        learning_rate=params["learning_rate"]
    )

    save_lstm_model(trained_model)
    predictions, mse, mae = evaluate_lstm(trained_model, X_test, y_test)
    print(f"LSTM MSE: {mse:.4f}, MAE: {mae:.4f}")
    plot_predictions(y_test, predictions, title="LSTM Predictions")


if __name__ == "__main__":
    model_type = "lstm"  # Change to "arima" to run ARIMA pipeline

    if model_type == "arima":
        run_arima_pipeline()
    elif model_type == "lstm":
        run_lstm_pipeline()
    else:
        print("Invalid model type. Choose 'arima' or 'lstm'.")