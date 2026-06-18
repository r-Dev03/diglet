mod model;
use js_sys::wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};
use leptos::html::Canvas;
use leptos::{prelude::*, svg::view, html};
use candle_core::{Device, Tensor, DType, Result, display};
use leptos::logging::log;
use leptos::prelude::*;



#[component]
fn App() -> impl IntoView {
  let canvas_ref: NodeRef<Canvas> = NodeRef::new();
  let (ctx, set_ctx) = signal(None);

  Effect::new(move |_| {
    // Get context
    if let Some(canvas) = canvas_ref.get() {
    // Store it 
      set_ctx.set(Some(canvas.get_context("2d")));
    }
  });


  view! {
    <canvas 
    node_ref=canvas_ref
    on:mousemove=move |ev| {
    if let Some(context) = ctx.get() {
    let context = context
        .unwrap()
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap();

    // use context
}

    // Wall
    // ctx.strokeRect(75, 140, 150, 110);

    // Door
    // ctx.fillRect(130, 190, 40, 60);

    // Roof
    // ctx.beginPath();
    // ctx.moveTo(50, 140);
    // ctx.lineTo(150, 60);
    // ctx.lineTo(250, 140);
    // ctx.closePath();
    // ctx.stroke();
    // Retrieve context from ???
    // Draw with it

    }

    // on:mousedown=move |ev| log!("Down at: {}, {}", ev.x(), ev.y()) on:mouseup=move |ev| log!("Up at: {}, {}", ev.x(), ev.y())
    // on:mousemove=move |ev| log!("Move: {}, {}", ev.x(), ev.y())
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
