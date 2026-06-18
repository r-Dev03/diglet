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
    if let Some(canvas) = canvas_ref.get() {
      if let Ok(Some(ctx)) = canvas.get_context("2d") {
        if let Ok(ctx) = ctx.dyn_into::<CanvasRenderingContext2d>() {
          set_ctx.set(Some(ctx));  
        }
      }
    }
  });

  view! {
  <canvas 
  node_ref=canvas_ref
  on:mousemove=move |ev| {
    if let Some(context) = ctx.get() {
      context.stroke_rect(75.0, 140.0, 150.0, 110.0);

      context.fill_rect(130.0, 190.0, 40.0, 60.0);

      context.begin_path();
      context.move_to(50.0, 140.0);
      context.line_to(150.0, 60.0);
      context.line_to(250.0, 140.0);
      context.close_path();
      context.stroke();

    }

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
