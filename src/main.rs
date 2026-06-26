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
  let (is_drawing, set_is_drawing) = signal(false);
  let (coordinates, set_coordinates) = signal((0, 0));

  Effect::new(move |_| {
    if let Some(canvas) = canvas_ref.get() {
      if let Ok(Some(ctx)) = canvas.get_context("2d") {
        if let Ok(ctx) = ctx.dyn_into::<CanvasRenderingContext2d>() {
          ctx.set_fill_style_str("white");
          ctx.fill_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);
          set_ctx.set(Some(ctx));  
        }
      }
    }
  });

  view! {
  <canvas 
  node_ref=canvas_ref
  on:mousedown=move |ev| { 
    set_coordinates.set((ev.x(), ev.y()));
    set_is_drawing.set(true); 

  }

  on:mouseup=move |ev| { 
    set_is_drawing.set(false); 
    if let Some(context) = ctx.get() {
      let image = context.get_image_data(0.0, 0.0, 500.0, 500.0);
      let mut greyscale: Vec<f32> = Vec::new();
      let mut image_data = &image.unwrap().data();

      let (chunks, _rest) = image_data.as_chunks::<4>();
      for (i, &[r, g, b, a]) in chunks.iter().take(10).enumerate() {
        let grey = f32::from(r) * 0.299 + f32::from(g) * 0.587 + f32::from(b) * 0.114;
        log!("pixel {} => r={}, g={}, b={}, a={}, grey={}", i, r, g, b, a, grey);
}

      // let (chunks, _rest) = image_data.as_chunks::<4>();
      // log!("{:?}", chunks);
      //
      // for &[r, g, b, a] in chunks {
      //   let grey = f32::from(r) * 0.299 + f32::from(g) * 0.587 + f32::from(b) * 0.114;
      //   let normalized = (grey / 255.0);
      //   greyscale.push(normalized);
      // }
      // log!("{:?}", greyscale);
      //
      //
      //
      //
      //
      // for i in image_data.chunks(4) {
      //   let grey = i[0] as f64 * 0.299 + i[1] as f64 * 0.587 + i[2] as f64 * 0.114;
      //   greyscale.push(grey);
      // }
      // for (i, v) in image_data.iter().enumerate().step_by(4) {
        // Gray=(R×0.299)+(G×0.587)+(B×0.114)
        // let grey = image_data[i] as f64 * 0.299 + image_data[i+1] as f64 * 0.587 + image_data[i+2] as f64 * 0.114;
        // greyscale.push(grey);
      // }
    }
  }

  on:mousemove=move |ev| {
    if let Some(context) = ctx.get() {

      if is_drawing.get() == true {
        context.begin_path();

        context.set_stroke_style(&wasm_bindgen::JsValue::from_str("black"));
        context.set_line_width(4.0);

        context.move_to(coordinates.get().0 as f64, coordinates.get().1 as f64);
        context.line_to(ev.x() as f64, ev.y() as f64);

        context.stroke();
      }

      set_coordinates.set((ev.x(), ev.y()));
    }

  }

    width="500"
      height="500"
      style="border: 1px solid black;"
      class="container"
    />
      <button
      on:click=move |ev| {
        if let Some(context) = ctx.get() {
          context.clear_rect(0.0, 0.0, 500.0, 500.0);
          set_coordinates.set((0, 0));
        } 
      }
      >
      <span class="button_top"> Clear Canvas </span>
    </button>

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
