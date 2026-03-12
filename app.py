import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="Credit Card Fraud Detection API")

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}


@app.post("/predict")
def predict(transaction: dict):

    # Convert input to dataframe
    data = pd.DataFrame([transaction])

    # Scale amount and time
    data["Amount"] = scaler.transform(data["Amount"].values.reshape(-1,1))
    data["Time"] = scaler.transform(data["Time"].values.reshape(-1,1))

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    result = "Fraud" if prediction == 1 else "Not Fraud"

    return {
        "prediction": result,
        "fraud_probability": float(probability)
    }