import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def load_model(path="models/lstm_model.pth"):
    model = LSTMModel()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_price(model, sequence):
    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # shape: (1, 60, 1)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction
