from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from config import CLEAN_TRAIN_PATH, CLEAN_TEST_PATH, BEST_MODEL_PATH
from save_model import save_model

# Load processed data
train_df = pd.read_csv(CLEAN_TRAIN_PATH)  
test_df = pd.read_csv(CLEAN_TEST_PATH)

# Split train data into features and target
X = train_df.drop(columns=["SalePrice"])
y = train_df["SalePrice"]

# Split train data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define models
models = {
      "Linear Regression": LinearRegression(),
      "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
      "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
      "LightGBM": LGBMRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = []
for model_name, model in models.items():
      print(f"🏃 Training {model_name}...")
      model.fit(X_train, y_train)
      y_pred = model.predict(X_val)
      
      # Evaluate model
      rmse = np.sqrt(mean_squared_error(y_val, y_pred))
      results.append({"model": model_name, "rmse": rmse})
      
      print(f"📏 Validation RMSE: {rmse:.2f}")
      
# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.sort_values(by="rmse", inplace=True)
print(f"Model Performance Comparison:\n{results_df}")

# Find the best model
best_model_name = results_df.iloc[0]["model"]
best_model = models[best_model_name]

# Use save_model function
save_model(best_model, BEST_MODEL_PATH)
    

