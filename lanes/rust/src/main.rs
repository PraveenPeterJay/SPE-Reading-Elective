use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use std::{fs, sync::Arc, time::Instant};

#[derive(Deserialize)]
struct PredictRequest { features: Vec<f32> }

#[derive(Serialize)]
struct PredictResponse {
    prediction:  i32,
    probability: f64,
    lane:        &'static str,
    runtime:     &'static str,
    latency_ms:  f64,
}

#[derive(Serialize)]
struct HealthResponse { status: &'static str, lane: &'static str }

#[derive(Deserialize)]
struct ScalerMeta { mean_: Vec<f32>, scale_: Vec<f32> }

struct AppState { session: Session, mean: Vec<f32>, scale: Vec<f32> }

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok", lane: "rust" })
}

async fn predict(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, StatusCode> {
    if req.features.len() != 8 { return Err(StatusCode::UNPROCESSABLE_ENTITY); }

    let t0 = Instant::now();

    let normed: Vec<f32> = req.features.iter().enumerate()
        .map(|(i, &v)| (v - state.mean[i]) / state.scale[i])
        .collect();

    let arr = Array2::from_shape_vec((1, 8), normed)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let input_tensor = Tensor::from_array(arr)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let outputs = state.session
        .run(ort::inputs!["features" => input_tensor]
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let logit_view = outputs["logit"]
        .try_extract_tensor::<f32>()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let logit = logit_view.view().iter().next()
        .copied()
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)? as f64;

    let prob = 1.0 / (1.0 + (-logit).exp());
    let pred = if prob >= 0.5 { 1 } else { 0 };
    let ms   = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(PredictResponse {
        prediction:  pred,
        probability: (prob * 1_000_000.0).round() / 1_000_000.0,
        lane:        "rust",
        runtime:     "ort 2.0.0-rc.10 (load-dynamic)",
        latency_ms:  (ms * 1000.0).round() / 1000.0,
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();

    let session = Session::builder()?
        .with_model_from_file("/app/model.onnx")?;

    let raw: ScalerMeta = serde_json::from_str(
        &fs::read_to_string("/app/scaler.json")?
    )?;

    let state = Arc::new(AppState {
        session,
        mean:  raw.mean_,
        scale: raw.scale_,
    });

    let app = Router::new()
        .route("/health",  get(health))
        .route("/predict", post(predict))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8501").await?;
    tracing::warn!("Rust lane listening on :8501");
    axum::serve(listener, app).await?;
    Ok(())
}