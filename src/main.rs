mod model;
use leptos::{prelude::*, svg::view};
use candle_core::{Device, Tensor, DType, Result, display};
use leptos::logging::log;

use leptos::prelude::*;

// Composing different components together is how we build
// user interfaces. Here, we'll define a reusable <ProgressBar/>.
// You'll see how doc comments can be used to document components
// and their properties.



#[component]
fn App() -> impl IntoView {
    view! {
        <canvas 
            on:mousedown=move |ev| log!("Down at: {}, {}", ev.x(), ev.y())
            on:mouseup=move |ev| log!("Up at: {}, {}", ev.x(), ev.y())
            on:mousemove=move |ev| log!("Move: {}, {}", ev.x(), ev.y())
            width="280"
            height="280"
            style="border: 1px solid black;"
        />
    }
}

fn main() {
  mount_to_body(App);
}



fn draw() {
  match model::load_weights("assets/weights.safetensors") {
    Ok(weights) => {
      println!("✓ Weights loaded successfully!");

      // Draw a vertical line (crude "1")
      let mut pixels = vec![0.0f32; 784];  // Note: f32
      for row in 5..23 {
        pixels[row * 28 + 14] = 1.0;
      }


      let input = Tensor::from_slice(&pixels, &[1, 784], &Device::Cpu).unwrap();

      match model::model_forward(&weights, &input) {
        Ok(output) => {
          println!("✓ Inference worked! {:?}", output);

          let values = output.to_vec2::<f32>().unwrap();
          println!("\nPredictions:");
          for (i, prob) in values[0].iter().enumerate() {
            println!("  Digit {}: {:.4}", i, prob);
          }
        }
        Err(e) => eprintln!("✗ Inference failed: {}", e),
      }
    }
    Err(e) => {
      eprintln!("✗ Failed to load weights: {}", e);
    }
  }
}
