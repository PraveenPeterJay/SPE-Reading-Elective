use spin_sdk::http::{IntoResponse, Request, Response, Method};
use spin_sdk::http_component;
use tract_onnx::prelude::*;
use image::io::Reader as ImageReader;
use image::imageops::FilterType;
use image::GenericImageView;
use std::io::Cursor;
use once_cell::sync::Lazy;

// THE PRO MOVE: Load the model once and keep it in static memory
// This variable is initialized only on the FIRST request.
static MODEL: Lazy<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>> = Lazy::new(|| {
    println!("Initializing Model (This should only happen once)...");
    tract_onnx::onnx()
        .model_for_path("mobilenetv2.onnx").unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224))).unwrap()
        .into_optimized().unwrap()
        .into_runnable().unwrap()
});

#[http_component]
fn handle_request(req: Request) -> anyhow::Result<impl IntoResponse> {
    match *req.method() {
        Method::Post => predict(req),
        _ => Ok(Response::builder().status(405).body("Method Not Allowed").build()),
    }
}

fn predict(req: Request) -> anyhow::Result<Response> {
    let body = req.body();
    let img_reader = ImageReader::new(Cursor::new(body)).with_guessed_format()?;
    let img = img_reader.decode()?;

    let resized = img.resize_exact(224, 224, FilterType::CatmullRom);

    // Optimized Single-Pass Pixel Loop
    let mut input_data = Vec::with_capacity(1 * 3 * 224 * 224);
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    // Note: Tract/MobileNet expects Planar RGB (All R, then all G, then all B)
    for c in 0..3 {
        for y in 0..224 {
            for x in 0..224 {
                let pixel = resized.get_pixel(x, y);
                let val = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                input_data.push(val);
            }
        }
    }

    // Access the static model (No disk read or optimization happens here!)
    let model = &*MODEL;

    let image_tensor = tract_ndarray::Array4::from_shape_vec((1, 3, 224, 224), input_data)?;
    let result = model.run(tvec!(image_tensor.into_tensor().into()))?;

    let output_view = result[0].to_array_view::<f32>()?;
    let scores = output_view.as_slice().unwrap();

    let (best_class, best_score) = scores.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &s)| (i, s))
        .unwrap();

    let json = serde_json::json!({
        "class": best_class,
        "score": best_score,
        "runtime": "Spin + Tract (Lazy Static)"
    });

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(json.to_string())
        .build())
}