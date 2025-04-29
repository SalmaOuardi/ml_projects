import torch

def save_model(model, path):
      torch.save(model.state_dict(), path)
      
      
def load_model(model, path, device):
      model = model_class().to(device)
      model.load_state_dict(torch.load(path))
      return model