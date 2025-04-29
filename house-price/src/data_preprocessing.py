import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew
import joblib
from config import TRAIN_PATH, TEST_PATH, CLEAN_TRAIN_PATH, CLEAN_TEST_PATH


def load_data():
    """Load raw train & test data."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def handle_missing_values(train_df, test_df):
    """Fill missing values: numerical with median, categorical with mode."""

    for col in train_df.columns:
        if train_df[col].isnull().sum() > 0:
            if train_df[col].dtype == "object":  # Categorical
                mode_value = train_df[col].mode()[0]
                train_df[col].fillna(mode_value, inplace=True)
                test_df[col].fillna(mode_value, inplace=True)
            else:  # Numerical
                median_value = train_df[col].median()
                train_df[col].fillna(median_value, inplace=True)
                test_df[col].fillna(median_value, inplace=True)

    return train_df, test_df

def transform_skewed_features(train_df, test_df, threshold=0.5):
    """Apply log transformation to skewed numerical features (if skewness > threshold)."""
    
    numeric_features = train_df.select_dtypes(include=["int64", "float64"]).columns
    skewness = train_df[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    
    skewed_features = skewness[abs(skewness) > threshold].index
    print(f"🔹 Transforming {len(skewed_features)} skewed features...")

    # Apply log1p transformation
    train_df[skewed_features] = np.log1p(train_df[skewed_features])
    test_df[skewed_features] = np.log1p(test_df[skewed_features])

    return train_df, test_df

def encode_categorical_features(train_df, test_df):
    """Convert categorical variables using One-Hot Encoding."""
    categorical_features = train_df.select_dtypes(include=["object"]).columns

    # One-Hot Encode
    train_df = pd.get_dummies(train_df, columns=categorical_features, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)

    # Ensure both train & test have the same columns
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    return train_df, test_df

def feature_selection(train_df):
    """Remove features with low correlation to SalePrice."""
    correlation = train_df.corr()["SalePrice"].sort_values(ascending=False)
    low_correlation_features = correlation[abs(correlation) < 0.1].index.tolist()

    print(f"🔹 Removing {len(low_correlation_features)} low-correlation features...")
    train_df.drop(columns=low_correlation_features, inplace=True)

    return train_df

def save_clean_data(train_df, test_df):
    """Save processed train & test data."""
    train_df.to_csv(CLEAN_TRAIN_PATH, index=False)
    test_df.to_csv(CLEAN_TEST_PATH, index=False)
    print("✅ Cleaned data saved!")

def preprocess_data():
    """Complete data preprocessing pipeline."""
    train_df, test_df = load_data()
    
    train_df, test_df = handle_missing_values(train_df, test_df)
    train_df, test_df = transform_skewed_features(train_df, test_df)
    train_df, test_df = encode_categorical_features(train_df, test_df)
    train_df = feature_selection(train_df)  # Only applied to train

    save_clean_data(train_df, test_df)

if __name__ == "__main__":
    preprocess_data()
