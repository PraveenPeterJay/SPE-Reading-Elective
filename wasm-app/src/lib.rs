use spin_sdk::http::{IntoResponse, Request, Response, Method};
use spin_sdk::http_component;
use tract_onnx::prelude::*;
use tract_onnx::prelude::tract_ndarray::Array2;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Instant;

// ─────────────────────────────────────────────
// 1. Scaler & Metadata — Loaded dynamically
// ─────────────────────────────────────────────
struct Scaler {
    mean: Vec<f32>,
    scale: Vec<f32>,
    n_features: usize,
}

static SCALER: Lazy<Scaler> = Lazy::new(|| {
    let raw = std::fs::read_to_string("scaler.json")
        .expect("Cannot open scaler.json. Ensure it is bundled in spin.toml");
    let v: Value = serde_json::from_str(&raw)
        .expect("Invalid scaler JSON");
    
    let mean: Vec<f32> = v["mean"].as_array()
        .expect("Scaler JSON must have 'mean' array")
        .iter().map(|x| x.as_f64().unwrap() as f32).collect();
        
    let scale: Vec<f32> = v["scale"].as_array()
        .expect("Scaler JSON must have 'scale' array")
        .iter().map(|x| x.as_f64().unwrap() as f32).collect();
    
    let n_features = mean.len();
    Scaler { mean, scale, n_features }
});

// ─────────────────────────────────────────────
// 2. Static model — Dynamically sized input
// ─────────────────────────────────────────────
type OnnxPlan = SimplePlan<
    TypedFact,
    Box<dyn TypedOp>,
    Graph<TypedFact, Box<dyn TypedOp>>,
>;

static MODEL: Lazy<OnnxPlan> = Lazy::new(|| {
    // We use the feature count from the scaler to set the input fact
    let n = SCALER.n_features as i64;

    tract_onnx::onnx()
        .model_for_path("model.onnx")
        .expect("Cannot open model.onnx")
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec!(1, n)),
        )
        .expect("Cannot set input fact")
        .into_optimized()
        .expect("Optimisation failed")
        .into_runnable()
        .expect("Cannot make runnable")
});

// ─────────────────────────────────────────────
// 3. Generic Request / Response types
// ─────────────────────────────────────────────
#[derive(Deserialize)]
struct PredictRequest {
    features: Vec<f32>,
}

#[derive(Deserialize)]
struct Sample {
    features: Vec<f32>,
    label: Option<i64>,
}

#[derive(Deserialize)]
struct BatchRequest {
    samples: Vec<Sample>,
}

#[derive(Serialize)]
struct PredictResponse {
    prediction: i64,
    confidence: f64,
    latency_ms: f64,
    runtime: String,
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
    results: Vec<SampleResult>,
    n_samples: usize,
    accuracy: Option<f64>,
    mean_latency_ms: f64,
    p99_latency_ms: f64,
    runtime: String,
}

// ─────────────────────────────────────────────
// 4. Pre-processing & Inference
// ─────────────────────────────────────────────
fn preprocess(features: &[f32]) -> anyhow::Result<Array2<f32>> {
    let scaler = &*SCALER;
    if features.len() != scaler.n_features {
        anyhow::bail!("Input dimension mismatch. Expected {}, got {}", scaler.n_features, features.len());
    }
    
    let scaled: Vec<f32> = features.iter().enumerate()
        .map(|(i, &v)| (v - scaler.mean[i]) / scaler.scale[i])
        .collect();
        
    Array2::from_shape_vec((1, scaler.n_features), scaled)
        .map_err(|e| anyhow::anyhow!("{}", e))
}

fn infer(features: &[f32]) -> anyhow::Result<(i64, f64, f64)> {
    let arr = preprocess(features)?;
    let tensor: Tensor = arr.into();

    let t0 = Instant::now();
    let outputs = MODEL.run(tvec!(tensor.into()))?;
    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Output 0: Class Label
    let label_view = outputs[0].to_array_view::<i64>()?;
    let label = label_view.as_slice().unwrap()[0];

    // Output 1: Probabilities (if available)
    let confidence: f64 = if outputs.len() > 1 {
        let prob_view = outputs[1].to_array_view::<f32>()?;
        let probs = prob_view.as_slice().unwrap();
        // If it's a binary classifier, return prob of class 1; else return prob of predicted class
        if probs.len() > label as usize {
            probs[label as usize] as f64
        } else {
            1.0 // Fallback if shape is unexpected
        }
    } else {
        1.0
    };

    Ok((label, confidence, latency_ms))
}

// ─────────────────────────────────────────────
// 5. Handlers
// ─────────────────────────────────────────────
#[http_component]
fn handle_request(req: Request) -> anyhow::Result<impl IntoResponse> {
    match (req.method(), req.path()) {
        (&Method::Post, "/predict") => handle_predict(req),
        (&Method::Post, "/batch")   => handle_batch(req),
        (&Method::Get,  "/health")  => handle_health(),
        _ => Ok(Response::builder().status(405).body("Not Found").build()),
    }
}

fn handle_predict(req: Request) -> anyhow::Result<Response> {
    let body: PredictRequest = serde_json::from_slice(req.body())?;
    let (label, conf, lat) = infer(&body.features)?;

    let resp = PredictResponse {
        prediction: label,
        confidence: (conf * 1e6).round() / 1e6,
        latency_ms: (lat  * 1e4).round() / 1e4,
        runtime: "wasm-spin-tract-generic".into(),
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&resp)?)
        .build())
}

fn handle_batch(req: Request) -> anyhow::Result<Response> {
    let body: BatchRequest = serde_json::from_slice(req.body())?;
    let mut results = Vec::new();
    let mut latencies = Vec::new();
    let mut correct = 0usize;

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

    let n = latencies.len() as f64;
    let mean_lat = latencies.iter().sum::<f64>() / n;
    let mut sorted = latencies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let resp = BatchResponse {
        results,
        n_samples: body.samples.len(),
        accuracy: if body.samples.iter().any(|s| s.label.is_some()) {
            Some((correct as f64 / body.samples.len() as f64 * 100.0).round() / 100.0)
        } else { None },
        mean_latency_ms: (mean_lat * 100.0).round() / 100.0,
        p99_latency_ms: (sorted[((n * 0.99) as usize).min(sorted.len() - 1)] * 100.0).round() / 100.0,
        runtime: "wasm-spin-tract-generic".into(),
    };

    Ok(Response::builder().status(200).header("content-type", "application/json").body(serde_json::to_string(&resp)?).build())
}

fn handle_health() -> anyhow::Result<Response> {
    let _ = &*MODEL;
    let _ = &*SCALER;
    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(format!(r#"{{"status":"ok","features_expected":{}}}"#, SCALER.n_features))
        .build())
}