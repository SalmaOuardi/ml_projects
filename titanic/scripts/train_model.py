import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from data_processing import process_data  # Ensure the latest cleaned data is used

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models):
    """Trains multiple models and evaluates them."""

    results = []

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)  # Train the model
        y_pred = model.predict(X_test)

        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities instead of class labels
        logloss = log_loss(y_test, y_prob)
        print(f"Log Loss: {logloss:.4f}")
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Store results
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1,
            "Log Loss": logloss,
            "Precision": report["1"]["precision"],
            "Recall": report["1"]["recall"]
        })

        # Save model
        joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")
        print(f"✅ {name} model saved.")

    return pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)

def main():
    """Loads cleaned data, trains models, and compares performance."""

    # Process & load cleaned data
    process_data()  
    train_df = pd.read_csv("data/clean_train.csv")

    # Separate features and target
    X = train_df.drop(["Survived", "PassengerId"], axis=1)
    y = train_df["Survived"]

    # Define models BEFORE using them
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42)
    }

    # Train-test split (K-Fold Cross-Validation)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n🔹 Cross-Validation Results:")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy", n_jobs=-1)
        print(f"✅ {name} CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Split the data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate all models
    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test, models)

    # Display results
    print("\n🔹 Model Performance Comparison:")
    print(results_df)

    # Plot results
    plt.figure(figsize=(8, 5))
    results_df.set_index("Model")[["Accuracy", "F1 Score"]].plot(kind="bar", colormap="viridis", edgecolor="black")
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()
