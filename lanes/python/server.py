"""
SPE Python Inference Lane — FastAPI + ONNX Runtime
"""
import json
import time
import os
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Load model and scaler ─────────────────────────────────────────────────────

SESS    = ort.InferenceSession("/app/model.onnx",
           providers=["CPUExecutionProvider"])
INPUT   = SESS.get_inputs()[0].name   # "features"

with open("/app/scaler.json") as f:
    sc = json.load(f)
MEAN    = np.array(sc["mean_"],  dtype=np.float32)
SCALE   = np.array(sc["scale_"], dtype=np.float32)
NAMES   = sc["feature_names"]

# ── API ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="SPE Python Lane", version="1.0.0")

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_items=8, max_items=8,
        example=[6, 148, 72, 35, 0, 33.6, 0.627, 50])

class PredictResponse(BaseModel):
    prediction:  int
    probability: float
    lane:        str = "python"
    runtime:     str = f"onnxruntime {ort.__version__}"
    latency_ms:  float

@app.get("/health")
def health():
    return {"status": "ok", "lane": "python"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != 8:
        raise HTTPException(422, "Expected exactly 8 features")
    t0 = time.perf_counter()
    x  = (np.array(req.features, dtype=np.float32) - MEAN) / SCALE
    x  = x.reshape(1, -1)
    logit = SESS.run(None, {INPUT: x})[0][0]
    prob  = float(1 / (1 + np.exp(-logit)))
    pred  = int(prob >= 0.5)
    ms    = (time.perf_counter() - t0) * 1000
    return PredictResponse(prediction=pred, probability=round(prob, 6), latency_ms=round(ms, 3))