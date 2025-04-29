import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_processing import process_data
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


def fine_tune_random_forest(X_train, y_train):
    """Fine-tunes Random Forest using RandomizedSearchCV."""
    param_grid = {
        "n_estimators": [100, 300, 500, 800, 1000],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    rf = RandomForestClassifier(random_state=42)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=20, cv=kf, verbose=1, n_jobs=-1)
    rf_search.fit(X_train, y_train)

    print("\n✅ Best Random Forest Parameters:", rf_search.best_params_)
    return rf_search.best_estimator_

def fine_tune_lightgbm(X_train, y_train):
    """Fine-tunes LightGBM using RandomizedSearchCV while silencing warnings."""
    param_grid = {
        "num_leaves": [31, 50, 100, 150],  # More flexible leaf size
        "max_depth": [-1, 10, 20, 30, 40],  # Allow deeper trees
        "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],  # Vary learning rate
        "n_estimators": [100, 300, 500, 800, 1200],  # More boosting rounds
        "min_child_samples": [5, 10, 20],  # Allow smaller node sizes
        "subsample": [0.6, 0.8, 1.0],  # Enable row sampling
        "colsample_bytree": [0.6, 0.8, 1.0]  # Enable feature sampling
    }

    lgb = LGBMClassifier(verbosity=-1, random_state=42)  # 🔹 Silence warnings here
    lgb_search = RandomizedSearchCV(lgb, param_distributions=param_grid, n_iter=20, cv=5, verbose=1, n_jobs=-1)
    lgb_search.fit(X_train, y_train)

    print("\n✅ Best LightGBM Parameters:", lgb_search.best_params_)
    return lgb_search.best_estimator_


def main():
    """Loads cleaned data, fine-tunes the best models, and evaluates them."""

    # Process & load cleaned data
    process_data()  
    train_df = pd.read_csv("data/clean_train.csv")

    # Separate features and target
    X = train_df.drop(["Survived", "PassengerId"], axis=1)
    y = train_df["Survived"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔹 Fine-tune models
    print("\n🔍 Fine-tuning Random Forest...")
    best_rf = fine_tune_random_forest(X_train, y_train)

    print("\n🔍 Fine-tuning LightGBM...")
    best_lgb = fine_tune_lightgbm(X_train, y_train)

    # 🔹 Evaluate fine-tuned models
    models = {"Random Forest": best_rf, "LightGBM": best_lgb}
    results = []

    for name, model in models.items():
      print(f"\n🔹 Evaluating {name}...")
      y_pred = model.predict(X_test)
      y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for log loss

      accuracy = accuracy_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred)
      logloss = log_loss(y_test, y_prob)
      report = classification_report(y_test, y_pred, output_dict=True)

      results.append({
            "Model": name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Log Loss": logloss,  # ✅ New metric
            "Precision": report["1"]["precision"],
            "Recall": report["1"]["recall"]
      })
      
      # Save fine-tuned models
      joblib.dump(model, f"models/fine_tuned_{name.replace(' ', '_').lower()}.pkl")
      print(f"✅ Fine-tuned {name} model saved.")

    # Convert to DataFrame
    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)

    # Display results
    print("\n🔹 Fine-Tuned Model Comparison:")
    print(results_df)

    # Plot results
    plt.figure(figsize=(8, 5))
    results_df.set_index("Model")[["Accuracy", "F1 Score"]].plot(kind="bar", colormap="plasma", edgecolor="black")
    plt.title("Fine-Tuned Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()
