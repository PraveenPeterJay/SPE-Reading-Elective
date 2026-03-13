use spin_sdk::http::{IntoResponse, Request, Response, Method};
use spin_sdk::http_component;
use tract_onnx::prelude::*;
// tract re-exports ndarray internally — access Array2 through it, not a separate crate
use tract_onnx::prelude::tract_ndarray::Array2;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Instant;

// ─────────────────────────────────────────────
// Feature count for Pima dataset
// ─────────────────────────────────────────────
const N_FEATURES: usize = 8;

// ─────────────────────────────────────────────
// 1. Static model — loaded once on first request
//    tract compiles the ONNX graph to an optimised
//    runnable plan; subsequent calls skip all setup.
// ─────────────────────────────────────────────
type OnnxPlan = SimplePlan<
    TypedFact,
    Box<dyn TypedOp>,
    Graph<TypedFact, Box<dyn TypedOp>>,
>;

static MODEL: Lazy<OnnxPlan> = Lazy::new(|| {
    tract_onnx::onnx()
        .model_for_path("model.onnx")
        .expect("Cannot open model.onnx")
        // shape: (1, 8)  dtype: f32
        .with_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec!(1, N_FEATURES as i64),
            ),
        )
        .expect("Cannot set input fact")
        .into_optimized()
        .expect("Optimisation failed")
        .into_runnable()
        .expect("Cannot make runnable")
});

// ─────────────────────────────────────────────
// 2. Scaler — loaded once from bundled JSON
// ─────────────────────────────────────────────
struct Scaler {
    mean:  Vec<f32>,
    scale: Vec<f32>,
}

static SCALER: Lazy<Scaler> = Lazy::new(|| {
    let raw = std::fs::read_to_string("scaler.json")
        .expect("Cannot open scaler.json");
    let v: Value = serde_json::from_str(&raw)
        .expect("Invalid scaler JSON");
    let mean  = v["mean"].as_array().unwrap()
        .iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let scale = v["scale"].as_array().unwrap()
        .iter().map(|x| x.as_f64().unwrap() as f32).collect();
    Scaler { mean, scale }
});

// ─────────────────────────────────────────────
// 3. Request / Response types
// ─────────────────────────────────────────────
#[derive(Deserialize)]
struct PredictRequest {
    features: Vec<f32>,
}

#[derive(Deserialize)]
struct Sample {
    features: Vec<f32>,
    label:    Option<i64>,
}

#[derive(Deserialize)]
struct BatchRequest {
    samples: Vec<Sample>,
}

#[derive(Serialize)]
struct PredictResponse {
    prediction: i64,
    label_text: String,
    confidence: f64,
    latency_ms: f64,
    runtime:    String,
}

#[derive(Serialize)]
struct SampleResult {
    prediction: i64,
    confidence: f64,
    latency_ms: f64,
    true_label: Option<i64>,
}

#[derive(Serialize)]
struct BatchResponse {
    results:         Vec<SampleResult>,
    n_samples:       usize,
    accuracy:        Option<f64>,
    mean_latency_ms: f64,
    p99_latency_ms:  f64,
    std_latency_ms:  f64,
    runtime:         String,
}

// ─────────────────────────────────────────────
// 4. Pre-processing (StandardScaler)
// ─────────────────────────────────────────────
fn preprocess(features: &[f32]) -> anyhow::Result<Array2<f32>> {
    if features.len() != N_FEATURES {
        anyhow::bail!("Expected {} features, got {}", N_FEATURES, features.len());
    }
    let scaler = &*SCALER;
    let scaled: Vec<f32> = features.iter().enumerate()
        .map(|(i, &v)| (v - scaler.mean[i]) / scaler.scale[i])
        .collect();
    Array2::from_shape_vec((1, N_FEATURES), scaled)
        .map_err(|e| anyhow::anyhow!("{}", e))
}

// ─────────────────────────────────────────────
// 5. Core inference — returns (label, confidence, latency_ms)
//
//    tract sklearn-onnx output layout (ZipMap pipeline):
//      output[0] → int64 label  shape [1]
//      output[1] → f32 probs    shape [1, 2]  (p_class0, p_class1)
// ─────────────────────────────────────────────
fn infer(features: &[f32]) -> anyhow::Result<(i64, f64, f64)> {
    let arr = preprocess(features)?;
    let tensor: Tensor = arr.into();

    let t0 = Instant::now();
    let outputs = MODEL.run(tvec!(tensor.into()))?;
    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // label
    let label_view = outputs[0].to_array_view::<i64>()?;
    let label = label_view.as_slice().unwrap()[0];

    // confidence (p of positive class)
    let confidence: f64 = if outputs.len() > 1 {
        let prob_view = outputs[1].to_array_view::<f32>()?;
        let probs = prob_view.as_slice().unwrap();
        // probs layout: [p_class0, p_class1]
        if probs.len() >= 2 { probs[1] as f64 } else { label as f64 }
    } else {
        label as f64
    };

    Ok((label, confidence, latency_ms))
}

// ─────────────────────────────────────────────
// 6. HTTP router
// ─────────────────────────────────────────────
#[http_component]
fn handle_request(req: Request) -> anyhow::Result<impl IntoResponse> {
    match (req.method(), req.uri().path()) {
        (&Method::Post, "/predict") => handle_predict(req),
        (&Method::Post, "/batch")   => handle_batch(req),
        (&Method::Get,  "/health")  => handle_health(),
        _ => Ok(Response::builder()
            .status(405)
            .body("Method or path not allowed")
            .build()),
    }
}

// ─────────────────────────────────────────────
// 7. POST /predict
// ─────────────────────────────────────────────
fn handle_predict(req: Request) -> anyhow::Result<Response> {
    let body: PredictRequest = serde_json::from_slice(req.body())?;
    let (label, conf, lat) = infer(&body.features)?;

    let resp = PredictResponse {
        prediction: label,
        label_text: if label == 1 { "Diabetic".into() } else { "Non-Diabetic".into() },
        confidence: (conf * 1e6).round() / 1e6,
        latency_ms: (lat  * 1e4).round() / 1e4,
        runtime:    "wasm-spin-tract".into(),
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&resp)?)
        .build())
}

// ─────────────────────────────────────────────
// 8. POST /batch
// ─────────────────────────────────────────────
fn handle_batch(req: Request) -> anyhow::Result<Response> {
    let body: BatchRequest = serde_json::from_slice(req.body())?;
    if body.samples.is_empty() {
        return Ok(Response::builder().status(422).body("Empty samples list").build());
    }

    let mut results   = Vec::new();
    let mut latencies = Vec::<f64>::new();
    let mut correct   = 0usize;

    for s in &body.samples {
        let (pred, conf, lat) = infer(&s.features)?;
        latencies.push(lat);
        if let Some(lbl) = s.label { if pred == lbl { correct += 1; } }
        results.push(SampleResult {
            prediction: pred,
            confidence: (conf * 1e6).round() / 1e6,
            latency_ms: (lat  * 1e4).round() / 1e4,
            true_label: s.label,
        });
    }

    let n        = latencies.len() as f64;
    let mean_lat = latencies.iter().sum::<f64>() / n;
    let std_lat  = (latencies.iter().map(|&x| (x - mean_lat).powi(2)).sum::<f64>() / n).sqrt();
    let mut sorted = latencies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99_idx  = ((n * 0.99) as usize).min(sorted.len() - 1);
    let accuracy = if body.samples.iter().any(|s| s.label.is_some()) {
        Some((correct as f64 / body.samples.len() as f64 * 1e4).round() / 1e4)
    } else { None };

    let resp = BatchResponse {
        results,
        n_samples:       body.samples.len(),
        accuracy,
        mean_latency_ms: (mean_lat * 1e4).round() / 1e4,
        p99_latency_ms:  (sorted[p99_idx] * 1e4).round() / 1e4,
        std_latency_ms:  (std_lat  * 1e4).round() / 1e4,
        runtime:         "wasm-spin-tract".into(),
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&resp)?)
        .build())
}

// ─────────────────────────────────────────────
// 9. GET /health
// ─────────────────────────────────────────────
fn handle_health() -> anyhow::Result<Response> {
    // Force static init so health confirms model is loaded
    let _ = &*MODEL;
    let _ = &*SCALER;
    let body = serde_json::json!({
        "status":  "ok",
        "runtime": "wasm-spin-tract",
        "note":    "memory metrics not available in WASI sandbox — use docker stats"
    });
    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(body.to_string())
        .build())
}