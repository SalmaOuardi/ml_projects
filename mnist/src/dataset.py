from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src import config

def get_data_loaders():
      transform = transforms.ToTensor()
      
      train_data = datasets.MNIST(root=config.DATA_DIR, train=True, download=True, transform=transform)
      test_data = datasets.MNIST(root=config.DATA_DIR, train=False, download=True, transform=transform)
      
      train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
      test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)
      
      return train_loader, test_loader