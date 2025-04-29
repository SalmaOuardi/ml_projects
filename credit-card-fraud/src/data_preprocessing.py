import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.config import DATA_PATH, PREPROCESSING_CONFIG

def load_data(model_name):
    df = pd.read_csv(DATA_PATH)
    y = df['Class']
    X = df.drop(columns=['Class'])

    config = PREPROCESSING_CONFIG[model_name]
    amount = X['Amount']

    # Step 1: log transform if needed
    if config["log_amount"]:
        amount = np.log1p(amount)

    # Step 2: scale if needed
    if config["scale"]:
        scaler = StandardScaler()
        amount = scaler.fit_transform(amount.values.reshape(-1, 1))
    
    # Replace the original column
    X['Amount'] = amount

    # Optional: scale 'Time' or drop it
    X = X.drop(columns=['Time'])

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


