import joblib
import os
from config import BEST_MODEL_PATH, MODELS_DIR

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def save_model(model, filename=BEST_MODEL_PATH):
    """Save a trained model to a file."""
    joblib.dump(model, filename)
    print(f"✅ Model saved to: {filename}")

def load_model(filename=BEST_MODEL_PATH):
    """Load a trained model from a file."""
    model = joblib.load(filename)
    print(f"✅ Model loaded from: {filename}")
    return model
