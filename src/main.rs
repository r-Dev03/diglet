// use leptos::prelude::*;
//
// fn main() {
//   mount_to_body(App);
// }
//
// #[component]
// fn App() -> impl IntoView {
// }
//
//
mod model;
use candle_core::{Device, Tensor, DType, Result, display};

fn main() {
  match model::load_weights("assets/weights.safetensors") {
    Ok(weights) => {
      println!("✓ Weights loaded successfully!");

      let input = Tensor::zeros(&[1, 784], DType::F32, &Device::Cpu).unwrap();

      match model::model_forward(&weights, &input) {
        Ok(output) => println!("✓ Inference worked! {:?}", output),
        Err(e) => eprintln!("✗ Inference failed: {}", e),
      }
    }
    Err(e) => {
      eprintln!("✗ Failed to load weights: {}", e);
    }
  }
}
