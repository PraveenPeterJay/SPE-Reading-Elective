use spin_sdk::http::{IntoResponse, Request, Response, Method};
use spin_sdk::http_component;
use tract_onnx::prelude::*;
use tract_onnx::prelude::tract_ndarray::Array2;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Instant;

// ─────────────────────────────────────────────
// 1. Scaler & Class Mapping — loaded dynamically
// ─────────────────────────────────────────────
struct ModelAssets {
    mean: Vec<f32>,
    scale: Vec<f32>,
    n_features: usize,
    class_map: Option<HashMap<i64, String>>,
}

static ASSETS: Lazy<ModelAssets> = Lazy::new(|| {
    // Load Scaler
    let s_raw = std::fs::read_to_string("scaler.json").expect("Cannot open scaler.json");
    let s_json: Value = serde_json::from_str(&s_raw).expect("Invalid scaler JSON");
    let mean: Vec<f32> = s_json["mean"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let scale: Vec<f32> = s_json["scale"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let n_features = mean.len();

    // Load Classes (Optional)
    let class_map = std::fs::read_to_string("classes.json").ok().and_then(|raw| {
        let json: HashMap<String, String> = serde_json::from_str(&raw).ok()?;
        // Convert string keys "0" to integer 0 for matching model output
        Some(json.into_iter().map(|(k, v)| (k.parse::<i64>().unwrap_or(0), v)).collect())
    });

    ModelAssets { mean, scale, n_features, class_map }
});

// ─────────────────────────────────────────────
// 2. Static model — Initialized based on Scaler
// ─────────────────────────────────────────────
type OnnxPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

static MODEL: Lazy<OnnxPlan> = Lazy::new(|| {
    let n = ASSETS.n_features as i64;
    tract_onnx::onnx()
        .model_for_path("model.onnx")
        .expect("Cannot open model.onnx")
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, n)))
        .expect("Cannot set input fact")
        .into_optimized()
        .expect("Optimisation failed")
        .into_runnable()
        .expect("Cannot make runnable")
});

// ─────────────────────────────────────────────
// 3. Request / Response types
// ─────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(untagged)] // This is the magic: it tries to match the structure
pub enum BatchInput {
    Wrapped { samples: Vec<Sample> },
    Direct(Vec<Sample>),
}

#[derive(Deserialize)]
struct PredictRequest { features: Vec<f32> }

#[derive(Deserialize)]
struct Sample { features: Vec<f32>, label: Option<i64> }

#[derive(Deserialize)]
struct BatchRequest { samples: Vec<Sample> }

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
    runtime:         String,
}

// ─────────────────────────────────────────────
// 4. Pre-processing & Inference
// ─────────────────────────────────────────────
fn preprocess(features: &[f32]) -> anyhow::Result<Array2<f32>> {
    let n = ASSETS.n_features;
    if features.len() != n {
        anyhow::bail!("Expected {} features, got {}", n, features.len());
    }
    let scaled: Vec<f32> = features.iter().enumerate()
        .map(|(i, &v)| (v - ASSETS.mean[i]) / ASSETS.scale[i])
        .collect();
    Array2::from_shape_vec((1, n), scaled).map_err(|e| anyhow::anyhow!("{}", e))
}

fn infer(features: &[f32]) -> anyhow::Result<(i64, String, f64, f64)> {
    let arr = preprocess(features)?;
    let tensor: Tensor = arr.into();

    let t0 = Instant::now();
    let outputs = MODEL.run(tvec!(tensor.into()))?;
    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // 1. Get the raw logit (F32)
    let logit = outputs[0].to_array_view::<f32>()?.as_slice().unwrap()[0];

    // 2. Thresholding: Logit > 0 means Class 1, else Class 0
    let label: i64 = if logit > 0.0 { 1 } else { 0 };

    // 3. Sigmoid for Confidence: 1 / (1 + exp(-logit))
    let confidence = (1.0 / (1.0 + (-logit).exp())) as f64;

    // 4. Class Mapping
    let label_text = ASSETS.class_map.as_ref()
        .and_then(|m| m.get(&label).cloned())
        .unwrap_or_else(|| format!("Class {}", label));

    Ok((label, label_text, confidence, latency_ms))
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
        _ => Ok(Response::builder().status(405).body("Not allowed").build()),
    }
}

fn handle_predict(req: Request) -> anyhow::Result<Response> {
    let body: PredictRequest = serde_json::from_slice(req.body())?;
    let (label, text, conf, lat) = infer(&body.features)?;

    let resp = PredictResponse {
        prediction: label,
        label_text: text,
        confidence: (conf * 1e6).round() / 1e6,
        latency_ms: (lat  * 1e4).round() / 1e4,
        runtime:    "wasm-spin-tract-generic".into(),
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&resp)?)
        .build())
}

fn handle_batch(req: Request) -> anyhow::Result<Response> {
    // 1. Deserialize into the Enum
    let input: BatchInput = serde_json::from_slice(req.body())?;
    
    // 2. Extract into a local vector
    let samples = match input {
        BatchInput::Wrapped { samples } => samples,
        BatchInput::Direct(samples) => samples,
    };

    // Safety check: Don't process empty batches
    // Safety check: Don't process empty batches
    if samples.is_empty() {
        return Ok(Response::builder()
            .status(422)
            .header("content-type", "application/json")
            // Returning a JSON error is better for your t.sh script to parse
            .body(r#"{"error": "Empty samples list"}"#)
            .build());
    }

    let mut results = Vec::new();
    let mut latencies = Vec::new();
    let mut correct = 0usize;

    for s in &samples {
        let (pred, _text, conf, lat) = infer(&s.features)?;
        latencies.push(lat);
        
        if let Some(lbl) = s.label { 
            if pred == lbl as i64 { correct += 1; } 
        }
        
        results.push(SampleResult {
            prediction: pred,
            confidence: (conf as f64 * 1e6).round() / 1e6,
            latency_ms: (lat  * 1e4).round() / 1e4,
            true_label: s.label,
        });
    }

    let n = latencies.len() as f64;
    let mean_lat = latencies.iter().sum::<f64>() / n;
    let mut sorted = latencies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // 3. Construct response using 'samples' and 'results' instead of 'body'
    let resp = BatchResponse {
        n_samples: samples.len(), // FIXED: replaced body.samples
        accuracy: if samples.iter().any(|s| s.label.is_some()) { // FIXED: replaced body.samples
            Some((correct as f64 / n * 1e4).round() / 1e4)
        } else { None },
        results,
        mean_latency_ms: (mean_lat * 1e4).round() / 1e4,
        p99_latency_ms: (sorted[((n * 0.99) as usize).min(sorted.len() - 1)] * 1e4).round() / 1e4,
        runtime: "wasm-spin-tract-generic".into(),
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&resp)?)
        .build())
}

fn handle_health() -> anyhow::Result<Response> {
    let _ = &*MODEL; // Ensure init
    let body = serde_json::json!({
        "status": "ok",
        "n_features": ASSETS.n_features,
        "runtime": "wasm-spin-tract-generic"
    });
    Ok(Response::builder().status(200).header("content-type", "application/json").body(body.to_string()).build())
}