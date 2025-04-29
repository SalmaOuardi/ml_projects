from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier

def get_model(name):
      if name == "logistic":
            return LogisticRegression(class_weight="balanced", random_state=42)
      elif name == "random_forest":
            return RandomForestClassifier(class_weight="balanced", random_state=42)
      elif name == "xgboost":
            return XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric="logloss", random_state=42)
      elif name == "isolation":
            return IsolationForest(contamination=0.001, random_state=42)
      else:
            raise ValueError(f"Model {name} not found")