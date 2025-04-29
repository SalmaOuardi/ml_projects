import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns

def check_data_leakage(df):
    """Checks for potential data leakage by finding high correlations with 'Survived'."""

    # Select only numerical features for correlation analysis
    numeric_df = df.select_dtypes(include=["number"])

    # Ensure 'Survived' is in the dataframe before correlation
    if "Survived" not in numeric_df.columns:
        print("❌ Error: 'Survived' column missing from data.")
        return

    # Compute correlation matrix
    correlations = numeric_df.corr()["Survived"].sort_values(ascending=False)
    print("\n🔹 Correlation of Numerical Features with 'Survived':\n", correlations)

    # Identify features with high correlation (possible data leakage)
    leaky_features = correlations[correlations > 0.9].index.tolist()
    if len(leaky_features) > 1:
        print(f"⚠️ Possible data leakage detected: {leaky_features[1:]} (Highly correlated with 'Survived')")
        
def compare_train_test_distribution(train_df, test_df, columns):
    """Compares feature distributions in train vs test data."""
    for column in columns:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(train_df[column], label="Train", shade=True)
        sns.kdeplot(test_df[column], label="Test", shade=True)
        plt.title(f"Distribution of {column} in Train vs Test")
        plt.legend()
        plt.show()

def extract_title(name):
    """Extracts title from passenger's name."""
    title = name.split(",")[1].split(".")[0].strip()
    common_titles = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master"}
    return common_titles.get(title, "Other")  # Map rare titles to "Other"

def clean_data(df):
    """Cleans Titanic dataset by handling missing values and adding new features."""
    
    # 🔹 Drop unnecessary columns
    df = df.drop(["Ticket", "Cabin"], axis=1, errors="ignore")

    # 🔹 Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())  # Median for Age
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # Mode for Embarked
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())  # Median for missing Fare (important for test set)

    # 🔹 Feature Engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Title"] = df["Name"].apply(extract_title)  # Extract title from Name
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # 🔹 Age Groups
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 50, 80], labels=["Child", "Teen", "Adult", "Senior"])

    # 🔹 Encode categorical features
    label_encoder = LabelEncoder()
    df["Sex"] = label_encoder.fit_transform(df["Sex"])  # Convert 'male'/'female' to 0/1
    df["Embarked"] = label_encoder.fit_transform(df["Embarked"])  # C, Q, S → Numeric
    df["Title"] = label_encoder.fit_transform(df["Title"])  # Convert titles to numbers
    df["AgeGroup"] = label_encoder.fit_transform(df["AgeGroup"])  # Encode age group

    # 🔹 Drop unnecessary columns after feature extraction
    df = df.drop(columns=["Name", "SibSp", "Parch"], errors="ignore")

    return df

def check_data_integrity(df):
    """Checks for data leaks, incorrect labels, and missing values."""
    
    print("\n🔍 Checking for Data Issues...\n")

    # 🔹 Check for missing values
    missing_values = df.isnull().sum()
    print("🔹 Missing Values per Column:\n", missing_values[missing_values > 0])

    # 🔹 Check for data leakage - Correlation with Survived
    if "Survived" in df.columns:
        df_numeric = df.select_dtypes(include=["number"])  # Select only numerical columns
        correlations = df_numeric.corr()["Survived"].sort_values(ascending=False)
        print("\n🔹 Correlation of Features with 'Survived':\n", correlations)

    # 🔹 Check for impossible values
    print("\n🔹 Checking for Impossible Values:")
    if "Age" in df.columns and (df["Age"] > 100).any():
        print("⚠️ Warning: Some passengers have an age > 100")
    if "Fare" in df.columns and (df["Fare"] == 0).sum() > 0:
        print(f"⚠️ Warning: {df[df['Fare'] == 0].shape[0]} passengers have Fare = 0")

    # 🔹 Check for unexpected categories
    categorical_columns = ["Sex", "Embarked", "Pclass"]
    for col in categorical_columns:
        if col in df.columns:
            print(f"\n🔹 Unique values in {col}: {df[col].unique()}")


def check_class_imbalance(df, target_column="Survived"):
    """Checks and visualizes class imbalance in the dataset."""
    
    class_counts = df[target_column].value_counts(normalize=True) * 100  # Convert to percentage
    print("\n🔹 Class Distribution:\n", class_counts)

    # Plot class distribution
    plt.figure(figsize=(6, 4))
    class_counts.plot(kind="bar", color=["royalblue", "darkorange"])
    plt.xticks([0, 1], labels=["Did Not Survive (0)", "Survived (1)"], rotation=0)
    plt.ylabel("Percentage of Passengers")
    plt.title("⚖️ Class Imbalance - Titanic Dataset")
    plt.show()

def check_naive_models(X_test, y_test):
    """Evaluates performance of simple baseline models"""
    
    # 1️⃣ Predict All "Did Not Survive" (Majority Class)
    y_naive = np.zeros(len(y_test))
    acc_naive = accuracy_score(y_test, y_naive)
    print(f"\n✅ Naïve Model (All Died) Accuracy: {acc_naive:.4f}")

    # 2️⃣ Predict Based on Gender
    if "Sex" in X_test.columns:
        y_naive_gender = (X_test["Sex"] == 1).astype(int)  # Females Survive
        acc_gender = accuracy_score(y_test, y_naive_gender)
        print(f"✅ Naïve Model (Based on Gender) Accuracy: {acc_gender:.4f}")

    # 3️⃣ Predict Based on Passenger Class
    if "Pclass" in X_test.columns:
        y_naive_pclass = (X_test["Pclass"] == 1).astype(int)  # Only 1st Class Survives
        acc_pclass = accuracy_score(y_test, y_naive_pclass)
        print(f"✅ Naïve Model (First Class Survives) Accuracy: {acc_pclass:.4f}")

def process_data():
    """Loads, cleans, and splits Titanic dataset while running EDA checks."""

    # Load raw data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    # Check data integrity before modifications
    check_data_integrity(train_df)
    
    #Check data leakage
    check_data_leakage(train_df)
    
    #Compare feature distributions(Train vs Test)
    #compare_train_test_distribution(train_df, test_df, ["Fare", "Age", "FamilySize"])

    # Clean data
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # Save cleaned datasets
    train_df.to_csv("data/clean_train.csv", index=False)
    test_df.to_csv("data/clean_test.csv", index=False)
    print("\n✅ Data cleaning completed and saved!")

    # Check class imbalance
    check_class_imbalance(train_df)

    # Prepare data for naïve model checks
    X = train_df.drop(["Survived", "PassengerId"], axis=1)
    y = train_df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run naïve model checks
    check_naive_models(X_test, y_test)

if __name__ == "__main__":
    process_data()
