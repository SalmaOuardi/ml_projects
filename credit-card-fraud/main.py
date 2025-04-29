from src.config import MODELS_TO_TRAIN
from src.data_preprocessing import load_data
from src.model_zoo import get_model
from src.train_model import train_model
from src.evaluate_model import evaluate_model

for model_name in MODELS_TO_TRAIN:
    print(f"\n🔧 Running pipeline for: {model_name}")
    X_train, X_test, y_train, y_test = load_data(model_name)
    model = get_model(model_name)
    model = train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test, model_name)
