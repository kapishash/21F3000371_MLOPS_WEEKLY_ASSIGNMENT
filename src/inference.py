from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

MODEL_PATH = "models/model.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(title="IRIS Prediction API")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"message": "IRIS Prediction API is running."}


@app.post("/predict")
def predict(data: IrisInput):

    prediction = model.predict([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    return {
        "prediction": prediction[0]
    }