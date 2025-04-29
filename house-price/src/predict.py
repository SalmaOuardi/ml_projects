import pandas as pd
import numpy as np
import os
from save_model import load_model
from config import CLEAN_TEST_PATH, BEST_MODEL_PATH, RESULTS_DIR

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load processed test data
test_df = pd.read_csv(CLEAN_TEST_PATH)

# Load the best trained model
best_model = load_model(BEST_MODEL_PATH)

# Ensure the correct test IDs are preserved
if "Id" in test_df.columns:
    test_ids = test_df["Id"]  # ✅ Keep the original test set IDs
else:
    raise ValueError("Error: 'Id' column is missing from test_df!")  # 🚨 Stop execution if missing

# Drop the ID column before prediction
test_df = test_df.drop(columns=["Id"], errors="ignore")  

# Make predictions
predictions = best_model.predict(test_df)

# Reverse the log transformation if applied to SalePrice
final_predictions = np.expm1(predictions)

# Save submission file dynamically
submission_path = os.path.join(RESULTS_DIR, "submission.csv")
submission = pd.DataFrame({"Id": test_ids, "SalePrice": final_predictions})

# ✅ Ensure IDs are sorted correctly before saving
submission = submission.sort_values("Id")

# ✅ Save with correct format
submission.to_csv(submission_path, index=False)

print(f"✅ Predictions saved to: {submission_path}")
