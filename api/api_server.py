# api_server.py
# -------------------------------

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# If you'd like to load your models here, you can (e.g. TFLite or Keras)...

app = FastAPI()


class ForecastRequest(BaseModel):
    recent_usage: list  # last 7 days usage


@app.post("/predict_consumption")
def predict_consumption(req: ForecastRequest):
    """
    A placeholder endpoint that would load a trained forecaster 
    and return a next-day usage prediction.
    """
    # For demo, just return an average:
    result = sum(req.recent_usage) / len(req.recent_usage)
    return {"next_day_forecast": result}
