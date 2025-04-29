import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
from config import PROCESSED_DATA_PATH, MODELS_DIR, BEST_MODEL_PATH
from sklearn.datasets import load_iris

# ✅ Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target  # Target variable

# ✅ Train-test split
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features ONLY for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define simple models
models = {
    "Logistic Regression": LogisticRegression(), 
    "Decision Tree": DecisionTreeClassifier(),  
    "SVM": svm.SVC(),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=3)
}

# Train & Evaluate Models
for name, model in models.items():
    if name == ["Logistic Regression","SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {name} Accuracy: {acc:.4f}")
