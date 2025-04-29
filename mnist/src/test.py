import torch

def evaluate(model, test_loader, device):
      model.eval()
      correct = 0
      with torch.no_grad():
            for inputs, labels in test_loader:
                  inputs, labels = inputs.to(device), labels.to(device)
                  outputs = model(inputs)
                  correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
      print(f'Test Accuracy: {correct / len(test_loader.dataset)}')