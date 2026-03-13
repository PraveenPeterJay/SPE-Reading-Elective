use spin_sdk::http::{IntoResponse, Request, Response, Method};
use spin_sdk::http_component;
use tract_onnx::prelude::*;
use tract_onnx::prelude::tract_ndarray::Array2;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Instant;

// ─────────────────────────────────────────────
// 1. Static model — Generic loading
// ─────────────────────────────────────────────
type OnnxPlan = SimplePlan<
    TypedFact,
    Box<dyn TypedOp>,
    Graph<TypedFact, Box<dyn TypedOp>>,
>;

static MODEL: Lazy<OnnxPlan> = Lazy::new(|| {
    // We load the model without forcing an input fact first, 
    // then let tract optimize it based on the file content.
    tract_onnx::onnx()
        .model_for_path("model.onnx")
        .expect("Cannot open model.onnx")
        .into_optimized()
        .expect("Optimisation failed")
        .into_runnable()
        .expect("Cannot make runnable")
});

// ─────────────────────────────────────────────
// 2. Scaler — Generic Key Handling
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
    
    // Support 'mean'/'mean_' and 'scale'/'std'
    let mean_json = v.get("mean").or(v.get("mean_")).expect("No mean key");
    let scale_json = v.get("scale").or(v.get("std")).expect("No scale key");

    let mean = mean_json.as_array().unwrap()
        .iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let scale = scale_json.as_array().unwrap()
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
    confidence: f64,
    latency_ms: f64,
    runtime:    String,
    features_processed: usize,
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
    runtime:         String,
}

// ─────────────────────────────────────────────
// 4. Pre-processing (Dynamic Feature Count)
// ─────────────────────────────────────────────
fn preprocess(features: &[f32]) -> anyhow::Result<Array2<f32>> {
    let scaler = &*SCALER;
    let expected = scaler.mean.len();
    
    if features.len() != expected {
        anyhow::bail!("Model expects {} features, got {}", expected, features.len());
    }
    
    let scaled: Vec<f32> = features.iter().enumerate()
        .map(|(i, &v)| (v - scaler.mean[i]) / scaler.scale[i])
        .collect();
        
    Array2::from_shape_vec((1, expected), scaled)
        .map_err(|e| anyhow::anyhow!("{}", e))
}

// ─────────────────────────────────────────────
// 5. Core inference
// ─────────────────────────────────────────────
fn infer(features: &[f32]) -> anyhow::Result<(i64, f64, f64)> {
    let arr = preprocess(features)?;
    let expected_count = SCALER.mean.len();
    
    // Explicitly set the shape for tract tensor conversion
    let tensor = tract_ndarray::Array::from_shape_vec((1, expected_count), arr.into_raw_vec())?.into();

    let t0 = Instant::now();
    let outputs = MODEL.run(tvec!(tensor))?;
    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // label extraction
    let label_view = outputs[0].to_array_view::<i64>()?;
    let label = label_view.as_slice().unwrap()[0];

    // confidence extraction
    let confidence: f64 = if outputs.len() > 1 {
        let prob_view = outputs[1].to_array_view::<f32>()?;
        let probs = prob_view.as_slice().unwrap();
        if probs.len() > label as usize { probs[label as usize] as f64 } else { 1.0 }
    } else {
        1.0
    };

    Ok((label, confidence, latency_ms))
}

// ─────────────────────────────────────────────
// 6. HTTP router
// ─────────────────────────────────────────────
#[http_component]
fn handle_request(req: Request) -> anyhow::Result<impl IntoResponse> {
    match (req.method(), req.path()) {
        (&Method::Post, "/predict") => handle_predict(req),
        (&Method::Post, "/batch")   => handle_batch(req),
        (&Method::Get,  "/health")  => handle_health(),
        _ => Ok(Response::builder()
            .status(405)
            .body("Method or path not allowed")
            .build()),
    }
}

fn handle_predict(req: Request) -> anyhow::Result<Response> {
    let body: PredictRequest = serde_json::from_slice(req.body())?;
    let (label, conf, lat) = infer(&body.features)?;

    let resp = PredictResponse {
        prediction: label,
        confidence: (conf * 1e6).round() / 1e6,
        latency_ms: (lat  * 1e4).round() / 1e4,
        runtime:    "wasm-spin-tract".into(),
        features_processed: SCALER.mean.len(),
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&resp)?)
        .build())
}

fn handle_batch(req: Request) -> anyhow::Result<Response> {
    let body: BatchRequest = serde_json::from_slice(req.body())?;
    if body.samples.is_empty() {
        return Ok(Response::builder().status(422).body("Empty samples list").build());
    }

    let mut results = Vec::new();
    let mut latencies = Vec::new();
    let mut correct = 0usize;

    for s in &body.samples {
        if let Ok((pred, conf, lat)) = infer(&s.features) {
            latencies.push(lat);
            if let Some(lbl) = s.label { if pred == lbl { correct += 1; } }
            results.push(SampleResult {
                prediction: pred,
                confidence: (conf * 1e6).round() / 1e6,
                latency_ms: (lat  * 1e4).round() / 1e4,
                true_label: s.label,
            });
        }
    }

    let n = latencies.len() as f64;
    let mean_lat = if n > 0.0 { latencies.iter().sum::<f64>() / n } else { 0.0 };

    let resp = BatchResponse {
        results,
        n_samples: body.samples.len(),
        accuracy: if n > 0.0 { Some(correct as f64 / n) } else { None },
        mean_latency_ms: (mean_lat * 1e4).round() / 1e4,
        runtime: "wasm-spin-tract".into(),
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&resp)?)
        .build())
}

fn handle_health() -> anyhow::Result<Response> {
    let _ = &*MODEL;
    let _ = &*SCALER;
    let body = serde_json::json!({
        "status":  "ok",
        "runtime": "wasm-spin-tract",
        "expected_features": SCALER.mean.len(),
        "note":    "memory metrics not available in WASI sandbox"
    });
    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(body.to_string())
        .build())
}