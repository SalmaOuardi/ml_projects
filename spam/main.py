from src.train_model import train_model
from src.evaluate_model import evaluate_model


def main():
      print("🚀 Starting Spam Classifier Training Pipeline...\n")
      
      #Train model and get test data
      X_test, y_test = train_model()
      
      #Evaluate the model
      evaluate_model(X_test, y_test)
      
      print("\n🎉 Training Pipeline Completed.")
      
      
if __name__ == "__main__":
      main()      