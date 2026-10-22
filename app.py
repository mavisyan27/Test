from fastapi.middleware.cors import CORSMiddleware

import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).with_name("diabetes_v12_histgradientboosting.joblib")
MODEL_VERSION = "v1.2"
FEATURES = ["age", "hypertension", "heart_disease", "bmi", "hba1c_level", "glucose"]
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Diabetes v1.2 Research Prediction API", version=MODEL_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    bmi: float = Field(..., gt=0, le=100)
    hba1c_level: float = Field(..., gt=0, le=30)
    glucose: float = Field(..., gt=0, le=1000)

class PredictionResponse(BaseModel):
    probability: float
    prediction: Literal[0, 1]
    risk_level: Literal["Low", "Moderate", "High"]
    model_version: str
    model_type: str
    warning: str

def check_api_key(authorization: str | None):
    expected = os.getenv("MODEL_API_KEY")
    if expected:
        supplied = (authorization or "")
        if supplied != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Invalid API key")

def risk_level(probability: float) -> str:
    if probability < 0.20:
        return "Low"
    if probability <= 0.50:
        return "Moderate"
    return "High"

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "model_type": "HistGradientBoostingClassifier"}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, authorization: str | None = Header(default=None)):
    check_api_key(authorization)
    values = np.array([[payload.age, payload.hypertension, payload.heart_disease,
                        payload.bmi, payload.hba1c_level, payload.glucose]], dtype=float)
    probability = float(model.predict_proba(values)[0, 1])
    prediction = int(probability >= 0.5)
    return PredictionResponse(
        probability=round(probability, 6),
        prediction=prediction,
        risk_level=risk_level(probability),
        model_version=MODEL_VERSION,
        model_type="HistGradientBoostingClassifier trained on output_data.xlsx",
        warning="Research and educational result only; not a medical diagnosis and not for insulin or medication decisions.",
    )

@app.post("/public-predict", response_model=PredictionResponse)
def public_predict(payload: PredictionRequest):
    values = np.array([[
        payload.age,
        payload.hypertension,
        payload.heart_disease,
        payload.bmi,
        payload.hba1c_level,
        payload.glucose,
    ]], dtype=float)

    probability = float(model.predict_proba(values)[0, 1])
    prediction = int(probability >= 0.5)

    return PredictionResponse(
        probability=round(probability, 6),
        prediction=prediction,
        risk_level=risk_level(probability),
        model_version=MODEL_VERSION,
        model_type="HistGradientBoostingClassifier trained on output_data.xlsx",
        warning=(
            "Public educational research demonstration only; "
            "not a medical diagnosis and not for insulin or medication decisions."
        ),
    )
