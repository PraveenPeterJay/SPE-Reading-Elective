use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Mutex;
use std::time::Instant;
use std::fs;

// ─────────────────────────────────────────────
// 1. Shared State
// ─────────────────────────────────────────────
struct AppState {
    session: Mutex<Session>,
    scaler_mean:  Vec<f32>,
    scaler_scale: Vec<f32>,
    input_shape: Vec<usize>, // Dynamically stored
    input_name: String,      // Dynamically stored
}

// ─────────────────────────────────────────────
// 2. Request / Response shapes
// ─────────────────────────────────────────────
#[derive(Deserialize)]
struct PredictRequest {
    features: Vec<f32>,
}

#[derive(Deserialize)]
struct Sample {
    features:  Vec<f32>,
    label:     Option<i64>,
}

#[derive(Deserialize)]
struct BatchRequest {
    samples: Vec<Sample>,
}

#[derive(Serialize)]
struct PredictResponse {
    prediction:  i64,
    confidence:  f64,
    latency_ms:  f64,
    runtime:     String,
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
// 3. Pre-processing (Generic Shape)
// ─────────────────────────────────────────────
fn preprocess(features: &[f32], mean: &[f32], scale: &[f32], expected: usize)
    -> Result<Array2<f32>, String>
{
    if features.len() != expected {
        return Err(format!("Model expects {} features, got {}", expected, features.len()));
    }
    let scaled: Vec<f32> = features.iter().enumerate()
        .map(|(i, &v)| (v - mean[i]) / scale[i])
        .collect();
    
    Array2::from_shape_vec((1, expected), scaled)
        .map_err(|e| e.to_string())
}

// ─────────────────────────────────────────────
// 4. Run Generic Inference
// ─────────────────────────────────────────────
fn infer(
    state:    &AppState,
    features: &[f32],
) -> Result<(i64, f64, f64), String> {
    let mut session = state.session.lock().unwrap();
    let expected_count = state.scaler_mean.len();
    
    let arr = preprocess(features, &state.scaler_mean, &state.scaler_scale, expected_count)?;
    
    let t0 = Instant::now();

    // Use dynamic shape from state
    let tensor = Tensor::from_array((state.input_shape.clone(), arr.into_raw_vec()))
        .map_err(|e| e.to_string())?;

    let outputs = session.run(ort::inputs![&state.input_name => tensor])
        .map_err(|e| e.to_string())?;

    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Extract Label
    let (_, labels) = outputs[0]
        .try_extract_tensor::<i64>()
        .map_err(|e| e.to_string())?;
    let predicted = labels[0];

    // Generic Confidence Extraction
    let confidence = if outputs.len() > 1 {
        match outputs[1].try_extract_tensor::<f32>() {
            Ok((_, probs)) => {
                let p = probs.as_slice().unwrap();
                if p.len() > predicted as usize { p[predicted as usize] as f64 } else { 1.0 }
            }
            Err(_) => 1.0,
        }
    } else {
        1.0
    };

    Ok((predicted, confidence, latency_ms))
}

// ─────────────────────────────────────────────
// 5. Handlers
// ─────────────────────────────────────────────
#[post("/predict")]
async fn predict(
    body: web::Json<PredictRequest>,
    data: web::Data<AppState>,
) -> impl Responder {
    match infer(&data, &body.features) {
        Ok((label, conf, lat)) => HttpResponse::Ok().json(PredictResponse {
            prediction: label,
            confidence: (conf * 1e6).round() / 1e6,
            latency_ms: (lat  * 1e4).round() / 1e4,
            runtime:    "rust-onnx".into(),
            features_processed: data.scaler_mean.len(),
        }),
        Err(e) => HttpResponse::UnprocessableEntity().json(serde_json::json!({ "error": e })),
    }
}

#[post("/batch")]
async fn batch(
    body: web::Json<BatchRequest>,
    data: web::Data<AppState>,
) -> impl Responder {
    let mut results = Vec::new();
    let mut latencies = Vec::new();
    let mut correct = 0usize;

    for s in &body.samples {
        if let Ok((pred, conf, lat)) = infer(&data, &s.features) {
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

    HttpResponse::Ok().json(BatchResponse {
        results,
        n_samples: body.samples.len(),
        accuracy: if n > 0.0 { Some(correct as f64 / n) } else { None },
        mean_latency_ms: (mean_lat * 1e4).round() / 1e4,
        runtime: "rust-onnx".into(),
    })
}

#[get("/health")]
async fn health(data: web::Data<AppState>) -> impl Responder {
    let rss_kb: u64 = fs::read_to_string("/proc/self/status")
        .unwrap_or_default().lines()
        .find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok()).unwrap_or(0);

    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "runtime": "rust-onnx",
        "rss_mb": (rss_kb as f64 / 1024.0),
        "expected_features": data.scaler_mean.len()
    }))
}

// ─────────────────────────────────────────────
// 8. Main
// ─────────────────────────────────────────────
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| "/app/model_artifacts/model.onnx".into());
    let scaler_path = std::env::var("SCALER_PATH").unwrap_or_else(|_| "/app/model_artifacts/scaler.json".into());

    // 1. Robust Scaler Loading
    let scaler_raw: Value = serde_json::from_str(&fs::read_to_string(&scaler_path).expect("Read error"))
        .expect("JSON error");
    
    // Check for 'mean' OR 'mean_'
    let mean_val = scaler_raw.get("mean").or(scaler_raw.get("mean_")).expect("No mean key");
    let scale_val = scaler_raw.get("scale").or(scaler_raw.get("std")).expect("No scale key");

    let scaler_mean: Vec<f32> = mean_val.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let scaler_scale: Vec<f32> = scale_val.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();

    // 2. Load Session and Extract Metadata
    let session = Session::builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .commit_from_file(&model_path).unwrap();

    let input = &session.inputs[0];
    let input_name = input.name.clone();
    // Extract dynamic shape [Batch, Features]
    let input_shape: Vec<usize> = input.input_type.tensor_type().unwrap().1
        .iter().map(|&d| if d == -1 { 1 } else { d as usize }).collect();

    println!("Generic Rust Lane: Model expects {} features.", scaler_mean.len());

    let state = web::Data::new(AppState {
        session: Mutex::new(session),
        scaler_mean,
        scaler_scale,
        input_shape,
        input_name,
    });

    HttpServer::new(move || {
        App::new().app_data(state.clone()).service(predict).service(batch).service(health)
    })
    .bind(("0.0.0.0", 8000))?.run().await
}