use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
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
    class_map:    Option<HashMap<i64, String>>, // New: Map for class names
}

// ─────────────────────────────────────────────
// 2. Request / Response Shapes
// ─────────────────────────────────────────────
#[derive(Deserialize)]
struct PredictRequest { features: Vec<f32> }

#[derive(Serialize)]
struct PredictResponse {
    prediction:  i64,
    label_text:  String, // Human-readable name
    confidence:  f64,
    latency_ms:  f64,
    runtime:     String,
}

// ─────────────────────────────────────────────
// 3. Core Inference Logic (The Type-Agnostic Version)
// ─────────────────────────────────────────────
fn infer(
    state:    &AppState,
    features: &[f32],
) -> Result<(i64, String, f64, f64), String> {
    let n = state.scaler_mean.len();
    
    // Validate Input Dimensions
    if features.len() != n {
        return Err(format!("Expected {} features, got {}", n, features.len()));
    }

    // Preprocess: (val - mean) / scale
    let scaled: Vec<f32> = features.iter().enumerate()
        .map(|(i, &v)| (v - state.scaler_mean[i]) / state.scaler_scale[i])
        .collect();
    
    let arr = Array2::from_shape_vec((1, n), scaled).map_err(|e| e.to_string())?;
    let mut session = state.session.lock().unwrap();

    let t0 = Instant::now();
    let tensor = Tensor::from_array(([1usize, n], arr.into_raw_vec())).map_err(|e| e.to_string())?;
    let outputs = session.run(ort::inputs![tensor]).map_err(|e| e.to_string())?;
    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // 1. Extract the raw Logit (it is an F32)
    let logit = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?.1[0];

    // 2. Thresholding: Positive logit = Class 1
    let predicted = if logit > 0.0 { 1 } else { 0 };

    // 3. Confidence: Sigmoid activation
    let confidence = (1.0 / (1.0 + (-logit).exp())) as f64;

    // 4. Mapping
    let label_text = state.class_map.as_ref()
        .and_then(|m| m.get(&predicted).cloned())
        .unwrap_or_else(|| format!("Class {}", predicted));

    Ok((predicted, label_text, confidence, latency_ms))
}

// ─────────────────────────────────────────────
// 4. Handlers
// ─────────────────────────────────────────────
#[post("/predict")]
async fn predict(body: web::Json<PredictRequest>, data: web::Data<AppState>) -> impl Responder {
    match infer(&data, &body.features) {
        Ok((val, text, conf, lat)) => HttpResponse::Ok().json(PredictResponse {
            prediction: val,
            label_text: text,
            confidence: (conf * 1e4).round() / 1e4,
            latency_ms: (lat  * 1e4).round() / 1e4,
            runtime:    "rust-ort-generic".into(),
        }),
        Err(e) => HttpResponse::UnprocessableEntity().json(json!({ "error": e })),
    }
}

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().json(json!({ "status": "ok", "runtime": "rust-ort-generic" }))
}

// ─────────────────────────────────────────────
// 5. Main (Loading All Artifacts)
// ─────────────────────────────────────────────
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model_path  = std::env::var("MODEL_PATH").unwrap_or_else(|_| "/app/model_artifacts/model.onnx".into());
    let scaler_path = std::env::var("SCALER_PATH").unwrap_or_else(|_| "/app/model_artifacts/scaler.json".into());
    let classes_path = std::env::var("CLASSES_PATH").unwrap_or_else(|_| "/app/model_artifacts/classes.json".into());

    // 1. Load Scaler
    let s_raw = fs::read_to_string(&scaler_path).expect("Failed to read scaler.json");
    let s_json: Value = serde_json::from_str(&s_raw).expect("Invalid scaler JSON");
    let scaler_mean: Vec<f32> = s_json["mean"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let scaler_scale: Vec<f32> = s_json["scale"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();

    // 2. Load Class Map (Optional)
    // 2. Load Class Map (Optional)
    let class_map: Option<HashMap<i64, String>> = fs::read_to_string(&classes_path).ok().and_then(|raw| {
        let json: HashMap<String, String> = serde_json::from_str(&raw).ok()?;
        // FIX: Parse the string key into an i64 integer
        Some(json.into_iter()
            .map(|(k, v)| (k.parse::<i64>().unwrap_or(0), v))
            .collect())
    });

    // 3. Load Model
    let session = Session::builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .commit_from_file(&model_path).unwrap();

    let state = web::Data::new(AppState {
        session: Mutex::new(session),
        scaler_mean,
        scaler_scale,
        class_map,
    });

    println!("Server running on 0.0.0.0:8000 (Features: {})", state.scaler_mean.len());
    HttpServer::new(move || {
        App::new().app_data(state.clone()).service(predict).service(health)
    })
    .bind(("0.0.0.0", 8000))?
    .run()
    .await
}