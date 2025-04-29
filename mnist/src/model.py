import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [1, 28, 28] → [32, 13, 13]
        x = self.pool(F.relu(self.conv2(x)))   # [32, 13, 13] → [64, 5, 5]
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)