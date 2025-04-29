DATA_PATH = "data/raw/creditcard.csv"
MODEL_PATH = "models/credit_card_fraud_model.pkl"
MODELS_TO_TRAIN = ["logistic", "random_forest", "xgboost", "isolation"]
PREPROCESSING_CONFIG = {
    "logistic": {"log_amount": True, "scale": True},
    "random_forest": {"log_amount": False, "scale": False},
    "xgboost": {"log_amount": False, "scale": False},
    "isolation": {"log_amount": False, "scale": False}
}