import torch
import torch.nn as nn
import torch.optim as optim
from src.model import CNN
from src.dataset import get_data_loaders
from src.train import train
from src.test import evaluate
from src.utils import save_model
from src import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_data_loaders()
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()  # Ensure criterion is defined here
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

train(model, train_loader, config.NUM_EPOCHS, optimizer, device, criterion)
evaluate(model, test_loader, device)
save_model(model, config.MODEL_PATH)