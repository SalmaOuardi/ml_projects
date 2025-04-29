import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.data_preprocessing import get_train_test_data, load_and_clean_data
from src import config

def train_model():
      # Load and clean data
      df = load_and_clean_data(config.DATA_PATH)  # Ensure the function accepts this argument
      
      # Split into train/test data
      X_train, X_test, y_train, y_test = get_train_test_data()  # Remove the unnecessary argument
      
      # TF-IDF Vectorization
      vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9)
      X_train_tfidf = vectorizer.fit_transform(X_train)
      
      # Train the naive bayes model
      model = MultinomialNB(alpha=0.5)
      model.fit(X_train_tfidf, y_train)
      
      # Save the model and vectorizer
      joblib.dump(model, config.MODEL_PATH)
      joblib.dump(vectorizer, config.VECTORIZER_PATH)
      
      print("✅ Model and vectorizer saved.")
      
      return X_test, y_test