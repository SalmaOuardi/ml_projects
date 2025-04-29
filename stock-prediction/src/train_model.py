import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.arima.model import ARIMA


def train_arima_model(series, order):
    """
    Train an ARIMA model on the provided time series data.

    Parameters:
        series (pd.Series): The time series data.
        order (tuple): The (p, d, q) order of the ARIMA model.

    Returns:
        ARIMA: The trained ARIMA model.
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def save_arima_model(model, filename):
    """
    Save the trained ARIMA model to a file.

    Parameters:
        model (ARIMA): The trained ARIMA model.
        filename (str): The name of the file to save the model to.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
        

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_lstm(model, X_train, y_train, batch_size=32, epochs=20, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    return model


def save_lstm_model(model, path="models/lstm_model.pth"):
    torch.save(model.state_dict(), path)