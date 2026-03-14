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
app = FastAPI(title="Generic ONNX Inference – Python")

# ─────────────────────────────────────────────
# 2. Load Artifacts (Generic Logic)
# ─────────────────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH",   "/app/model_artifacts/model.onnx")
SCALER_PATH  = os.getenv("SCALER_PATH",  "/app/model_artifacts/scaler.json")
CLASSES_PATH = os.getenv("CLASSES_PATH", "/app/model_artifacts/classes.json")

print(f"Loading ONNX model from {MODEL_PATH} ...")
ort_session = ort.InferenceSession(MODEL_PATH)
INPUT_NAME  = ort_session.get_inputs()[0].name

print(f"Loading scaler from {SCALER_PATH} ...")
with open(SCALER_PATH) as f:
    scaler_data = json.load(f)
SCALER_MEAN = np.array(scaler_data["mean"],  dtype=np.float32)
SCALER_STD  = np.array(scaler_data["scale"], dtype=np.float32)
N_FEATURES  = len(SCALER_MEAN)

print(f"Loading classes from {CLASSES_PATH} (optional) ...")
CLASS_MAP = {}
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH) as f:
        raw_classes = json.load(f)
        # FIX: Ensure keys are integers for matching model output
        CLASS_MAP = {int(k): v for k, v in raw_classes.items()}
else:
    print("No classes.json found, using default indices.")

print(f"Startup complete. Features expected: {N_FEATURES}")

# ─────────────────────────────────────────────
# 3. Helper Logic
# ─────────────────────────────────────────────
def preprocess(features: list[float]) -> np.ndarray:
    if len(features) != N_FEATURES:
        raise ValueError(f"Dimension mismatch: expected {N_FEATURES}, got {len(features)}")
    
    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    arr = (arr - SCALER_MEAN) / SCALER_STD
    return arr

def extract_label_and_conf(outputs):
    # outputs[0] is the "logit" [batch, 1]
    logit = float(outputs[0][0])
    
    # 1. Prediction: Threshold at 0.0 for Logits
    predicted_label = 1 if logit > 0 else 0
    
    # 2. Confidence: Sigmoid function to convert Logit -> Probability
    # formula: 1 / (1 + exp(-x))
    confidence = 1.0 / (1.0 + np.exp(-logit))
    
    return predicted_label, float(confidence)

# ─────────────────────────────────────────────
# 4. API Endpoints
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
        outputs      = ort_session.run(None, {INPUT_NAME: input_tensor})
        latency_ms   = (time.perf_counter() - t0) * 1000

        pred, conf = extract_label_and_conf(outputs)
        
        return {
            "prediction":  pred,
            "label_text":  CLASS_MAP.get(pred, f"Class {pred}"),
            "confidence":  round(conf, 6),
            "latency_ms":  round(latency_ms, 4),
            "runtime":     "python-onnx-generic",
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/batch")
async def batch_predict(request: Request):
    # Load the raw body
    body = await request.json()
    if isinstance(body, list):
        samples = body
    elif isinstance(body, dict):
        samples = body.get("samples", [])
    else:
        return JSONResponse({"error": "Invalid format. Provide a list of samples."}, status_code=422)

    if not samples:
        return JSONResponse({"error": "Batch is empty"}, status_code=422)

    results, latencies, correct = [], [], 0

    for s in samples:
        try:
            t0 = time.perf_counter()
            inp = preprocess(s["features"])
            outs = ort_session.run(None, {INPUT_NAME: inp})
            lat = (time.perf_counter() - t0) * 1000
            
            pred, conf = extract_label_and_conf(outs)
            true_label = s.get("label")
            
            latencies.append(lat)
            if true_label is not None and pred == int(true_label):
                correct += 1

            results.append({
                "prediction": pred,
                "confidence": round(conf, 6),
                "latency_ms": round(lat, 4),
                "true_label": true_label,
            })
        except Exception as e:
            return JSONResponse({"error": f"Sample error: {str(e)}"}, status_code=400)

    accuracy = correct / len(samples) if samples else None
    return {
        "results":          results,
        "n_samples":        len(samples),
        "accuracy":         round(accuracy, 4) if accuracy is not None else None,
        "mean_latency_ms":  round(float(np.mean(latencies)), 4),
        "p99_latency_ms":   round(float(np.percentile(latencies, 99)), 4),
        "runtime":          "python-onnx-generic",
    }

@app.get("/health")
async def health():
    proc = psutil.Process()
    return {
        "status":        "ok",
        "n_features":    N_FEATURES,
        "rss_mb":        round(proc.memory_info().rss / 1024 / 1024, 2),
        "class_mapping": CLASS_MAP
    }