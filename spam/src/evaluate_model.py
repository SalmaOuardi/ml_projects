import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.config import MODEL_PATH, VECTORIZER_PATH

def evaluate_model(X_test, y_test):
      
      #load saved model and vectorizer
      model = joblib.load(MODEL_PATH)
      vectorizer = joblib.load(VECTORIZER_PATH)
      
      #transform the test data
      X_test_tfidf = vectorizer.transform(X_test)
      
      #predict the test data
      y_pred = model.predict(X_test_tfidf)
      
      #calculate evaluation metrics
      print("✅ Evaluation Metrics:")
      print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
      print(f"Precision: {precision_score(y_test, y_pred):.4f}")
      print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
      print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
      print("\nConfusion Matrix:")
      print(confusion_matrix(y_test, y_pred))
      print("\nClassification Report:")
      print(classification_report(y_test, y_pred))