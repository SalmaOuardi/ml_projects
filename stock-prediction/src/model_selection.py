#from pmdarima import auto_arima
from typing import Tuple
"""
def find_best_arima_params(series: pd.Series, seasonal: bool = False) -> Tuple[int, int, int]:
    model = auto_arima(
          series, 
          seasonal=seasonal, 
          stepwise=True, 
          trace=True,
          error_action="ignore",
          suppress_warnings=True
          )
    return model.order
"""

def get_lstm_hyperparams():
    """
    Get default hyperparameters for PyTorch LSTM model.

    Returns:
        dict: Dictionary of training and model hyperparameters.
    """
    return {
        "window_size": 60,       # how many timesteps per input sequence
        "hidden_size": 50,       # LSTM units (same as 'units' before)
        "num_layers": 1,         # you can expand later
        "dropout": 0.2,
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 0.001,
    }
