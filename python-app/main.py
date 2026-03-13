import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import time
import os
import psutil
import resource

# ─────────────────────────────────────────────
# 1. App Initialisation
# ─────────────────────────────────────────────
app = FastAPI(title="Generic Inference Lane – Python/ONNX")

# ─────────────────────────────────────────────
# 2. Load Model & Scaler (Generic Detection)
# ─────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH",  "/app/model_artifacts/model.onnx")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/model_artifacts/scaler.json")

print(f"Loading ONNX model from {MODEL_PATH}...")
ort_session = ort.InferenceSession(MODEL_PATH)

# DYNAMIC METADATA EXTRACTION
input_meta     = ort_session.get_inputs()[0]
INPUT_NAME     = input_meta.name
INPUT_SHAPE    = input_meta.shape  # e.g., [Batch, Features]
EXPECTED_COUNT = INPUT_SHAPE[1]

print(f"Model loaded  | Input: {INPUT_NAME} | Expected Features: {EXPECTED_COUNT}")

print(f"Loading scaler from {SCALER_PATH}...")
with open(SCALER_PATH) as f:
    scaler = json.load(f)

# Safe Key Loading: Handles 'mean' vs 'mean_' and 'scale' vs 'std'
SCALER_MEAN = np.array(scaler.get("mean", scaler.get("mean_")), dtype=np.float32)
SCALER_STD  = np.array(scaler.get("scale", scaler.get("std", scaler.get("scale_"))), dtype=np.float32)

if len(SCALER_MEAN) != EXPECTED_COUNT:
    print(f"WARNING: Scaler size ({len(SCALER_MEAN)}) mismatch with Model features ({EXPECTED_COUNT})")

print("Scaler loaded successfully.")

# ─────────────────────────────────────────────
# 3. Generic Pre-processor
# ─────────────────────────────────────────────
def preprocess(features: list[float]) -> np.ndarray:
    """
    Validates and scales input features based on the detected model shape.
    """
    if len(features) != EXPECTED_COUNT:
        raise ValueError(f"Model expects {EXPECTED_COUNT} features, but received {len(features)}")
    
    # Convert to numpy and scale
    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    arr = (arr - SCALER_MEAN) / SCALER_STD
    return arr.astype(np.float32)

# ─────────────────────────────────────────────
# 4. /predict – single-sample inference
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    features = body.get("features")
    if features is None:
        return JSONResponse({"error": "Missing 'features' key"}, status_code=422)

    try:
        t0 = time.perf_counter()
        input_tensor = preprocess(features)
        
        # Run Inference
        outputs = ort_session.run(None, {INPUT_NAME: input_tensor})
        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract Prediction (Generic flattening)
        predicted_label = int(np.array(outputs[0]).flatten()[0])

        # Confidence Handling (Generic)
        confidence = 1.0
        if len(outputs) > 1:
            prob_raw = outputs[1]
            if isinstance(prob_raw, list) and len(prob_raw) > 0:
                # Handle sklearn-onnx list of dicts
                d = prob_raw[0]
                confidence = float(d.get(predicted_label, d.get(str(predicted_label), 0.5)))
            else:
                # Handle raw probability arrays
                prob_arr = np.array(prob_raw).flatten()
                if len(prob_arr) > predicted_label:
                    confidence = float(prob_arr[predicted_label])

        return {
            "prediction":  predicted_label,
            "confidence":  round(confidence, 6),
            "latency_ms":  round(latency_ms, 4),
            "runtime":     "python-onnx",
            "features_processed": EXPECTED_COUNT
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ─────────────────────────────────────────────
# 5. /batch – bulk inference
# ─────────────────────────────────────────────
@app.post("/batch")
async def batch_predict(request: Request):
    body = await request.json()
    samples = body.get("samples", [])
    if not samples:
        return JSONResponse({"error": "Empty 'samples' list"}, status_code=422)

    results, latencies, correct = [], [], 0

    for s in samples:
        try:
            features   = s["features"]
            true_label = s.get("label")

            t0   = time.perf_counter()
            inp  = preprocess(features)
            outs = ort_session.run(None, {INPUT_NAME: inp})
            ms   = (time.perf_counter() - t0) * 1000

            pred = int(np.array(outs[0]).flatten()[0])
            latencies.append(ms)
            
            if true_label is not None and pred == true_label:
                correct += 1

            results.append({
                "prediction": pred,
                "latency_ms": round(ms, 4),
                "true_label": true_label
            })
        except Exception:
            continue

    return {
        "n_samples":       len(samples),
        "mean_latency_ms": round(float(np.mean(latencies)), 4) if latencies else 0,
        "runtime":         "python-onnx",
        "results":         results
    }

# ─────────────────────────────────────────────
# 6. /health – resource snapshot
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    proc = psutil.Process()
    return {
        "status":        "ok",
        "runtime":       "python-onnx",
        "rss_mb":        round(proc.memory_info().rss / 1024 / 1024, 2),
        "expected_features": EXPECTED_COUNT,
        "model_inputs":  [i.name for i in ort_session.get_inputs()]
    }

# ─────────────────────────────────────────────
# 7. Metrics
# ─────────────────────────────────────────────
_counters = {"requests": 0, "errors": 0}

@app.middleware("http")
async def count_requests(request: Request, call_next):
    _counters["requests"] += 1
    return await call_next(request)

@app.get("/metrics")
async def metrics():
    return _counters