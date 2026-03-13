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
app = FastAPI(title="Diabetes Risk Inference – Python/ONNX")

# ─────────────────────────────────────────────
# 2. Load Model & Scaler (once at startup)
# ─────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH",  "/app/model_artifacts/model.onnx")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/model_artifacts/scaler.json")

print("Loading ONNX model …")
ort_session = ort.InferenceSession(MODEL_PATH)
INPUT_NAME  = ort_session.get_inputs()[0].name
print(f"Model loaded  |  input name: {INPUT_NAME}")

print("Loading scaler …")
with open(SCALER_PATH) as f:
    scaler = json.load(f)
SCALER_MEAN = np.array(scaler["mean"],  dtype=np.float32)
SCALER_STD  = np.array(scaler["scale"], dtype=np.float32)
print("Scaler loaded.")

# Feature names for the 8 Pima columns (for documentation / error messages)
FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ─────────────────────────────────────────────
# 3. Helper – pre-process raw feature vector
# ─────────────────────────────────────────────
def preprocess(features: list[float]) -> np.ndarray:
    """
    Accepts a list of 8 raw Pima feature values and returns a
    (1, 8) float32 numpy array scaled with the training scaler.
    """
    if len(features) != 8:
        raise ValueError(f"Expected 8 features, got {len(features)}")
    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    arr = (arr - SCALER_MEAN) / SCALER_STD
    return arr.astype(np.float32)

# ─────────────────────────────────────────────
# 4. /predict  –  single-sample inference
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(request: Request):
    """
    Body (JSON):
        { "features": [7, 159, 64, 29, 125, 27.4, 0.294, 40] }
    or the full test_samples.json row format:
        { "features": [...], "label": 0 }
    """
    body = await request.json()
    features = body.get("features")
    if features is None:
        return JSONResponse({"error": "Missing 'features' key"}, status_code=422)

    t0 = time.perf_counter()

    input_tensor = preprocess(features)
    outputs      = ort_session.run(None, {INPUT_NAME: input_tensor})

    latency_ms = (time.perf_counter() - t0) * 1000

    # Most sklearn-ONNX pipelines emit:
    #   outputs[0] → predicted label  (shape: [1])
    #   outputs[1] → probability dict or array
    predicted_label = int(outputs[0][0])

    # Probability of positive class (diabetes = 1)
    if len(outputs) > 1:
        prob_raw = outputs[1]
        if isinstance(prob_raw, list):          # list-of-dicts (sklearn-onnx style)
            confidence = float(prob_raw[0].get(1, prob_raw[0].get("1", 0.5)))
        else:                                   # plain ndarray [[p0, p1]]
            confidence = float(np.array(prob_raw)[0][1])
    else:
        confidence = float(predicted_label)

    return {
        "prediction":  predicted_label,
        "label_text":  "Diabetic" if predicted_label == 1 else "Non-Diabetic",
        "confidence":  round(confidence, 6),
        "latency_ms":  round(latency_ms, 4),
        "runtime":     "python-onnx",
    }

# ─────────────────────────────────────────────
# 5. /batch  –  bulk inference for benchmarking
# ─────────────────────────────────────────────
@app.post("/batch")
async def batch_predict(request: Request):
    """
    Body (JSON):
        { "samples": [ {"features": [...], "label": 0}, … ] }
    Returns per-sample predictions plus aggregate metrics.
    """
    body    = await request.json()
    samples = body.get("samples", [])
    if not samples:
        return JSONResponse({"error": "Empty 'samples' list"}, status_code=422)

    results   = []
    latencies = []
    correct   = 0

    for s in samples:
        features    = s["features"]
        true_label  = s.get("label")

        t0           = time.perf_counter()
        inp          = preprocess(features)
        outs         = ort_session.run(None, {INPUT_NAME: inp})
        latency_ms   = (time.perf_counter() - t0) * 1000

        pred = int(outs[0][0])
        if len(outs) > 1:
            pb = outs[1]
            conf = float(pb[0].get(1, 0.5)) if isinstance(pb, list) else float(np.array(pb)[0][1])
        else:
            conf = float(pred)

        latencies.append(latency_ms)
        if true_label is not None and pred == true_label:
            correct += 1

        results.append({
            "prediction": pred,
            "confidence": round(conf, 6),
            "latency_ms": round(latency_ms, 4),
            "true_label": true_label,
        })

    accuracy = correct / len(samples) if samples else None
    return {
        "results":          results,
        "n_samples":        len(samples),
        "accuracy":         round(accuracy, 4) if accuracy is not None else None,
        "mean_latency_ms":  round(float(np.mean(latencies)), 4),
        "p99_latency_ms":   round(float(np.percentile(latencies, 99)), 4),
        "std_latency_ms":   round(float(np.std(latencies)), 4),
        "runtime":          "python-onnx",
    }

# ─────────────────────────────────────────────
# 6. /health  –  liveness + resource snapshot
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    proc   = psutil.Process()
    mem_mb = proc.memory_info().rss / 1024 / 1024
    # Peak RSS (useful for cold-start memory profiling)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "status":        "ok",
        "runtime":       "python-onnx",
        "rss_mb":        round(mem_mb, 2),
        "peak_rss_kb":   peak_kb,
        "model_inputs":  [i.name for i in ort_session.get_inputs()],
        "model_outputs": [o.name for o in ort_session.get_outputs()],
    }

# ─────────────────────────────────────────────
# 7. /metrics  –  Prometheus-compatible counters
#    (lightweight; no prometheus_client dependency)
# ─────────────────────────────────────────────
_counters = {"requests": 0, "errors": 0}

@app.middleware("http")
async def count_requests(request: Request, call_next):
    _counters["requests"] += 1
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        _counters["errors"] += 1
        raise e

@app.get("/metrics")
async def metrics():
    return _counters