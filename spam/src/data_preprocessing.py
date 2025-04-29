import pandas as pd
from sklearn.model_selection import train_test_split
from src import config

def load_and_clean_data(data_path):
      #Load dataset
      df = pd.read_csv(data_path, encoding='latin-1')
      
      #Keep only the necessary columns
      df = df[['v1', 'v2']]
      df.columns = ['label', 'text']
      
      #map the labels to 0 and 1
      df['label'] = df['label'].map({'ham': 0, 'spam': 1})
      
      return df

def get_train_test_data(test_size=0.2, random_state=42):
    # Load and clean the data
    df = load_and_clean_data(config.DATA_PATH)
    
    # Ensure correct splitting of features and labels
    X = df['text']  # Features
    y = df['label']  # Labels
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


