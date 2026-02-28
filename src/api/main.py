import os
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- CONFIGURATION ---
"""Environment-driven configuration to avoid hard-coded secrets.

Values are provided via environment variables; sensible local defaults
are used when variables are not set.
"""
# 1. MLflow & MinIO connection (read from ENV with fallback)
_mlflow_s3_endpoint = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
_aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "admin")
_aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "password")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Ensure dependent libraries see the values via os.environ
os.environ["MLFLOW_S3_ENDPOINT_URL"] = _mlflow_s3_endpoint
os.environ["AWS_ACCESS_KEY_ID"] = _aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = _aws_secret_access_key

# 2. Registered model name in train_mlflow.py
MODEL_NAME = os.getenv("MODEL_NAME", "TelcoChurnModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "None")  # or "Production"
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

# Global store for loaded model
ml_models = {}

# --- DATA MODELS ---


class CustomerRequest(BaseModel):
    # Define required features for prediction
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    # Thêm các feature khác nếu cần


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- LOAD MODEL AT STARTUP ---
    print("🔌 Connecting to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # Load model version from MLflow (default 1, overridable via ENV)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        print(f"📥 Loading model from: {model_uri}")

        model = mlflow.sklearn.load_model(model_uri)
        ml_models["churn_model"] = model
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️ API will start but predictions are unavailable.")

    yield

    # Clean up
    ml_models.clear()

app = FastAPI(lifespan=lifespan, title="CDP Churn Prediction API")


@app.get("/")
def home():
    return {"message": "CDP API is running with MLflow integration 🚀"}


@app.post("/predict")
def predict_churn(customer: CustomerRequest):
    if "churn_model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([customer.dict()])

        # Predict
        model = ml_models["churn_model"]
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return {
            "prediction": int(prediction),
            "churn_probability": float(probability),
            "risk_level": "High" if probability > 0.7 else ("Medium" if probability > 0.4 else "Low")
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")
