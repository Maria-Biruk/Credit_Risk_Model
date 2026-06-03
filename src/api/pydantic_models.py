from pydantic import BaseModel


class PredictionInput(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float


class PredictionOutput(BaseModel):
    risk_probability: float
    prediction: int
