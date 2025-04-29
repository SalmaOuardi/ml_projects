import torch
import torch.nn as nn
from src import config

def train(model, train_loader, num_epochs, optimizer, device, criterion):
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # Ensure criterion is passed correctly
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Step {i + 1}, Loss: {loss.item()}")