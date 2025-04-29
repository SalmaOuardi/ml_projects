import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import RAW_DATA_PATH, FEATURES, PROCESSED_DATA_PATH

def load_data(path: str) -> pd.DataFrame:
    print(f"📥 Loading data from {path}...")
    df = pd.read_csv(path)
    df.drop(columns=['CustomerID'], inplace=True, errors='ignore')
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess(df: pd.DataFrame, features: list) -> pd.DataFrame:
    print(f"⚙️ Preprocessing features: {features}")
    X = df[features]
    scaled = StandardScaler().fit_transform(X)
    print("✅ Scaling complete.")
    return pd.DataFrame(scaled, columns=features)

def save_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Processed data saved to {path}")

def run_pipeline():
    print("🚀 Starting preprocessing pipeline...")
    df_raw = load_data(RAW_DATA_PATH)
    df_processed = preprocess(df_raw, FEATURES)
    save_data(df_processed, PROCESSED_DATA_PATH)
    print("🏁 Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()
