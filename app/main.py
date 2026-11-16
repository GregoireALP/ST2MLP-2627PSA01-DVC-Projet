import os
from pathlib import Path

import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model")) / "model"

model = mlflow.sklearn.load_model(str(MODEL_DIR))

app = FastAPI(title="Digits RandomForest")


class PredictRequest(BaseModel):
    # One or more 8x8 digit images, flattened to 64 values (0-16)
    instances: list[list[float]] = Field(..., examples=[[[0.0] * 64]])


@app.post("/predict")
def predict(req: PredictRequest):
    preds = model.predict(req.instances)
    return {"predictions": preds.tolist()}
