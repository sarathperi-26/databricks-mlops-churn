import os
from datetime import datetime, timezone

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")
THRESHOLD = 0.5

model = mlflow.sklearn.load_model(MODEL_PATH)
app = FastAPI(title="Telco Churn Model", version="1.0.0")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=1000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    frame = pd.DataFrame([c.model_dump() for c in request.customers])
    proba = model.predict_proba(frame)[:, 1]
    return {
        "model_version": MODEL_VERSION,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "predictions": [
            {"churn_probability": round(float(p), 4), "churn": int(p >= THRESHOLD)}
            for p in proba
        ],
    }
