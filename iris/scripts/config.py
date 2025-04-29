import os

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# File paths
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "iris.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "clean_iris.csv")
DATA_SUMMARY_PATH = os.path.join(RESULTS_DIR, "data_summary.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)
    

