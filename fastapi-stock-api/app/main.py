from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import load_model, predict_price

app = FastAPI()
model = load_model()  # Load once when the app starts

class StockInput(BaseModel):
    data: list[list[float]]  # shape (60, 1)

@app.get("/")
def root():
    return {"message": "FastAPI is live and ready for LSTM!"}

@app.post("/predict")
def predict(input: StockInput):
    sequence = input.data
    prediction = predict_price(model, sequence)
    return {"predicted_price": prediction}
