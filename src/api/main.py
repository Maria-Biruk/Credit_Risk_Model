import os
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from src.api.pydantic_models import PredictionInput, PredictionOutput

import mlflow

app = FastAPI()

# Resolve project root and common local model locations
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "model.pkl"
NOTEBOOK_MODEL_PATH = BASE_DIR / "notebooks" / "models" / "model.pkl"

# Prefer MLflow registered model if available, otherwise fall back to local joblib
model = None
mlflow_model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/credit-risk-model/latest")

try:
    # try loading model from MLflow registry (works when MLflow tracking is configured)
    model = mlflow.pyfunc.load_model(mlflow_model_uri)
except Exception:
    # fallback to local model path
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.exists():
        if NOTEBOOK_MODEL_PATH.exists():
            model_path = NOTEBOOK_MODEL_PATH
        else:
            raise FileNotFoundError(
                (
                    "Model not found. Checked MLflow URI '")
                + f"{mlflow_model_uri}' and local paths: {DEFAULT_MODEL_PATH}, "
                + f"{NOTEBOOK_MODEL_PATH}.\n"
                + (
                    "Set MLFLOW_MODEL_URI or MODEL_PATH environment variables, or "
                    "place model at root/models/model.pkl."
                )
            )

    model = joblib.load(model_path)


@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}


@app.post("/predict")
def predict(data: PredictionInput):
    input_df = pd.DataFrame([data.dict()])

    # mlflow.pyfunc models return a pyfunc wrapper that accepts DataFrame
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
        else:
            # some pyfunc models only implement predict; use predict as proxy
            pred_raw = model.predict(input_df)
            # if predict returns probabilities, try to extract
            if hasattr(pred_raw, "__len__") and len(pred_raw) and isinstance(pred_raw[0], (list, tuple)):
                prob = float(pred_raw[0][1])
            else:
                prob = float(pred_raw[0])

        pred = int(model.predict(input_df)[0])
    except Exception as e:
        # bubble up a clear error for debugging
        raise RuntimeError(f"Model prediction failed: {e}")

    return PredictionOutput(risk_probability=float(prob), prediction=int(pred)).dict()
