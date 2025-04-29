import joblib
import os

# Load vectorizer and model once
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")
MODEL_PATH = os.path.join("model", "spam_classifier.pkl") 

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)


def predict_spam(text):
    """
    Predict whether the given text is spam or not.
    Returns: ('Spam' or 'Ham', confidence score)
    """
    X = vectorizer.transform([text])
    prediction_proba = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]

    label = "Spam" if prediction == 1 else "Ham"
    confidence = round(prediction_proba[prediction], 3)

    return label, confidence
