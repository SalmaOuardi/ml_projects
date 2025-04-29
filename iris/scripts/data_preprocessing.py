import pandas as pd
import json
import os
from sklearn.preprocessing import LabelEncoder
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, DATA_SUMMARY_PATH

def preprocess_data():
    """Loads, cleans, and saves processed dataset while storing insights."""

    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    # Encode categorical target variable
    label_encoder = LabelEncoder()
    df["Species"] = label_encoder.fit_transform(df["Species"])

    # Compute dataset insights
    insights = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "feature_means": df.mean().to_dict(),
        "feature_std": df.std().to_dict(),
        "class_distribution": df["Species"].value_counts(normalize=True).to_dict()
    }

    # Save processed dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("✅ Data preprocessing complete!")

    # Ensure results directory exists
    os.makedirs(os.path.dirname(DATA_SUMMARY_PATH), exist_ok=True)

    # Save insights as JSON
    with open(DATA_SUMMARY_PATH, "w") as f:
        json.dump(insights, f, indent=4)

    print(f"✅ Data insights saved to {DATA_SUMMARY_PATH}")

# Run preprocessing
if __name__ == "__main__":
    preprocess_data()
