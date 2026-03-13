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
    label_text:  String,
    confidence:  f64,
    latency_ms:  f64,
    runtime:     String,
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
// 3. Pre-processing (apply StandardScaler)
// ─────────────────────────────────────────────
fn preprocess(features: &[f32], mean: &[f32], scale: &[f32])
    -> Result<Array2<f32>, String>
{
    if features.len() != 8 {
        return Err(format!("Expected 8 features, got {}", features.len()));
    }
    let scaled: Vec<f32> = features.iter().enumerate()
        .map(|(i, &v)| (v - mean[i]) / scale[i])
        .collect();
    // shape: (1, 8)
    Array2::from_shape_vec((1, 8), scaled)
        .map_err(|e| e.to_string())
}

// ─────────────────────────────────────────────
// 4. Run one inference, return (label, confidence, latency_ms)
// ─────────────────────────────────────────────
fn infer(
    session:  &mut Session,
    features: &[f32],
    mean:     &[f32],
    scale:    &[f32],
) -> Result<(i64, f64, f64), String> {
    let arr   = preprocess(features, mean, scale)?;
    let shape = [1usize, 8usize];

    let t0 = Instant::now();

    let tensor = Tensor::from_array((shape, arr.into_raw_vec()))
        .map_err(|e| e.to_string())?;

    let outputs = session.run(ort::inputs![tensor])
        .map_err(|e| e.to_string())?;

    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // outputs[0] → predicted label (int64, shape [1])
    let (_, labels) = outputs[0]
        .try_extract_tensor::<i64>()
        .map_err(|e| e.to_string())?;
    let predicted = labels[0];

    // outputs[1] → probabilities — sklearn-onnx emits a sequence of maps
    //   [{0: p0, 1: p1}, …]  serialised as raw f32 pairs in ORT
    //   Fallback: use label as confidence if output[1] unavailable
    let confidence = if outputs.len() > 1 {
        match outputs[1].try_extract_tensor::<f32>() {
            Ok((_, probs)) => {
                // flat layout: [p_class0, p_class1] for batch=1
                if probs.len() >= 2 { probs[1] as f64 } else { predicted as f64 }
            }
            Err(_) => predicted as f64,
        }
    } else {
        predicted as f64
    };

    Ok((predicted, confidence, latency_ms))
}

// ─────────────────────────────────────────────
// 5. POST /predict
// ─────────────────────────────────────────────
#[post("/predict")]
async fn predict(
    body: web::Json<PredictRequest>,
    data: web::Data<AppState>,
) -> impl Responder {
    let mut session = data.session.lock().unwrap();
    match infer(&mut session, &body.features, &data.scaler_mean, &data.scaler_scale) {
        Ok((label, conf, lat)) => HttpResponse::Ok().json(PredictResponse {
            prediction: label,
            label_text: if label == 1 { "Diabetic".into() } else { "Non-Diabetic".into() },
            confidence: (conf * 1e6).round() / 1e6,
            latency_ms: (lat  * 1e4).round() / 1e4,
            runtime:    "rust-onnx".into(),
        }),
        Err(e) => HttpResponse::UnprocessableEntity().json(
            serde_json::json!({ "error": e })
        ),
    }
}

// ─────────────────────────────────────────────
// 6. POST /batch
// ─────────────────────────────────────────────
#[post("/batch")]
async fn batch(
    body: web::Json<BatchRequest>,
    data: web::Data<AppState>,
) -> impl Responder {
    if body.samples.is_empty() {
        return HttpResponse::UnprocessableEntity()
            .json(serde_json::json!({ "error": "Empty samples list" }));
    }

    let mut session  = data.session.lock().unwrap();
    let mut results  = Vec::new();
    let mut latencies = Vec::new();
    let mut correct  = 0usize;

    for s in &body.samples {
        match infer(&mut session, &s.features, &data.scaler_mean, &data.scaler_scale) {
            Ok((pred, conf, lat)) => {
                latencies.push(lat);
                if let Some(lbl) = s.label { if pred == lbl { correct += 1; } }
                results.push(SampleResult {
                    prediction: pred,
                    confidence: (conf * 1e6).round() / 1e6,
                    latency_ms: (lat  * 1e4).round() / 1e4,
                    true_label: s.label,
                });
            }
            Err(e) => return HttpResponse::InternalServerError()
                .json(serde_json::json!({ "error": e })),
        }
    }

    let n   = latencies.len() as f64;
    let mean_lat = latencies.iter().sum::<f64>() / n;
    let std_lat  = (latencies.iter().map(|&x| (x - mean_lat).powi(2)).sum::<f64>() / n).sqrt();
    let mut sorted = latencies.clone(); sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99_idx  = ((n * 0.99) as usize).min(sorted.len() - 1);
    let accuracy = if body.samples.iter().any(|s| s.label.is_some()) {
        Some((correct as f64 / body.samples.len() as f64 * 1e4).round() / 1e4)
    } else { None };

    HttpResponse::Ok().json(BatchResponse {
        results,
        n_samples:       body.samples.len(),
        accuracy,
        mean_latency_ms: (mean_lat * 1e4).round() / 1e4,
        p99_latency_ms:  (sorted[p99_idx] * 1e4).round() / 1e4,
        std_latency_ms:  (std_lat  * 1e4).round() / 1e4,
        runtime:         "rust-onnx".into(),
    })
}

// ─────────────────────────────────────────────
// 7. GET /health
// ─────────────────────────────────────────────
#[get("/health")]
async fn health() -> impl Responder {
    // Read /proc/self/status for RSS without external crates
    let rss_kb: u64 = fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    HttpResponse::Ok().json(serde_json::json!({
        "status":    "ok",
        "runtime":   "rust-onnx",
        "rss_kb":    rss_kb,
        "rss_mb":    (rss_kb as f64 / 1024.0 * 100.0).round() / 100.0,
    }))
}

// ─────────────────────────────────────────────
// 8. Main
// ─────────────────────────────────────────────
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model_path  = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/app/model_artifacts/model.onnx".into());
    let scaler_path = std::env::var("SCALER_PATH")
        .unwrap_or_else(|_| "/app/model_artifacts/scaler.json".into());

    println!("Loading scaler from {scaler_path} …");
    let scaler_raw: Value = serde_json::from_str(
        &fs::read_to_string(&scaler_path).expect("Cannot read scaler.json")
    ).expect("Invalid scaler JSON");
    let scaler_mean: Vec<f32> = scaler_raw["mean"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let scaler_scale: Vec<f32> = scaler_raw["scale"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    println!("Scaler loaded ({} features).", scaler_mean.len());

    println!("Loading ONNX model from {model_path} …");
    let session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .commit_from_file(&model_path)
        .unwrap();
    println!("Model loaded. Starting server on 0.0.0.0:8000");

    let state = web::Data::new(AppState {
        session:      Mutex::new(session),
        scaler_mean,
        scaler_scale,
    });

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(web::JsonConfig::default().limit(1_048_576)) // 1 MB
            .service(predict)
            .service(batch)
            .service(health)
    })
    .bind(("0.0.0.0", 8000))?
    .run()
    .await
}