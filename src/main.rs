mod model;
mod preprocess;
use candle_core::{Device, Tensor};
use js_sys::wasm_bindgen::JsCast;
use leptos::html::Canvas;
use leptos::logging::log;
use leptos::prelude::*;
use web_sys::CanvasRenderingContext2d;

#[component]
fn App() -> impl IntoView {
  let canvas_ref: NodeRef<Canvas> = NodeRef::new();
  let (ctx, set_ctx) = signal(None);
  let (is_drawing, set_is_drawing) = signal(false);
  let (coordinates, set_coordinates) = signal((0, 0));
  let (prediction, set_prediction) = signal(String::from(""));

  Effect::new(move |_| {
    if let Some(canvas) = canvas_ref.get() {
      if let Ok(Some(ctx)) = canvas.get_context("2d") {
        if let Ok(ctx) = ctx.dyn_into::<CanvasRenderingContext2d>() {
          let window = web_sys::window().unwrap();
          let dpr = window.device_pixel_ratio();
          let display_size = canvas.get_bounding_client_rect().width();

          // set physical canvas internal resolution size
          canvas.set_width((display_size * dpr) as u32);
          canvas.set_height((display_size * dpr) as u32);

          // scale drawing context so standard layout coords map perfectly to high-res pixels
          ctx.scale(dpr, dpr).unwrap();
          ctx.set_fill_style_str("black");
          ctx.fill_rect(0.0, 0.0, display_size, display_size);
          set_ctx.set(Some(ctx));
        }
      }
    }
  });

  view! {
  <main class="app">
  <h1 class="logo">"D"<span class="logo-i">"ı"</span>"g"<span class="logo-one">"1"</span>"et"</h1>
  <div class="canvas-frame">
  <canvas class="canvas"
  node_ref=canvas_ref
  on:pointerdown=move |ev| { // pointer down -> enable drawing
    if let Some(canvas) = canvas_ref.get() {
      let _ = canvas.set_pointer_capture(ev.pointer_id());
      let rect = canvas.get_bounding_client_rect();
      let x = ev.client_x() as f64 - rect.left();
      let y = ev.client_y() as f64 - rect.top();
      set_coordinates.set((x as i32, y as i32));
      set_is_drawing.set(true);
    }
  }

  on:pointermove=move |ev| { // pointer move -> start drawing
    if let Some(canvas) = canvas_ref.get() {
      let rect = canvas.get_bounding_client_rect();
      let x = ev.client_x() as f64 - rect.left();
      let y = ev.client_y() as f64 - rect.top();

      if let Some(context) = ctx.get() {
        if is_drawing.get() {
          context.begin_path();
          context.set_stroke_style(&wasm_bindgen::JsValue::from_str("white"));
          context.set_line_width(rect.width() * 0.072);
          context.set_line_cap("round");
          context.set_line_join("round");
          context.move_to(coordinates.get().0 as f64, coordinates.get().1 as f64);
          context.line_to(x as f64, y as f64);
          context.stroke();
        }

        set_coordinates.set((x as i32, y as i32));
      }

    }
  }

  on:pointerup=move |_ev| { // pointer up -> stop drawing & and retrieve data
    set_is_drawing.set(false);
    if let Some(context) = ctx.get() {
      if let Some(canvas) = canvas_ref.get() {
        let width = canvas.width();


        let image = context.get_image_data(0.0, 0.0, width as f64, width as f64);
        let image_data = image.unwrap().data();

        let mut greyscale: Vec<f32> = Vec::new();
        let (chunks, _rest) = image_data.as_chunks::<4>();


        for &[r, g, b, _a] in chunks {
          let grey = f32::from(r) * 0.299 + f32::from(g) * 0.587 + f32::from(b) * 0.114;
          let normalized = grey / 255.0;
          greyscale.push(normalized);
        }

        match draw(greyscale, width as usize) {
          Ok(digit) => set_prediction.set(digit.to_string()),
          Err(e) => log!("inference failed: {e}")
        }

      }
    }

  }
      />
    </div>

      <div class="controls">
      <button on:click=move |_ev| {
        if let (Some(context), Some(canvas)) = (ctx.get(), canvas_ref.get()) {
          let size = canvas.get_bounding_client_rect().width();
          context.clear_rect(0.0, 0.0, size, size);
          context.set_fill_style_str("black");
          context.fill_rect(0.0, 0.0, size, size);
          set_coordinates.set((0, 0));
          set_prediction.set("".to_string());
        }
      }>
      <span class="button_top">"Clear Canvas"</span>
      </button>
      <span class="prediction">"Prediction: "<span class="prediction-digit">{move || prediction.get()}</span></span>
      </div>
      </main>
  }
}

fn draw(data: Vec<f32>, size: usize) -> Result<usize, String> {
  if data.len() != size * size {
    return Err(format!(
      "expected {} values for a {size}x{size} canvas, got {}",
      size * size,
      data.len()
    ));
  }

  let Some(small) = preprocess::preprocess(&data, size) else {
    return Err("nothing drawn yet".to_string());
  };

  let device = &Device::Cpu;
  let bytes = include_bytes!("../public/weights.safetensors");
  let weights = model::load_weights(bytes).map_err(|e| format!("failed to load weights: {e}"))?;

  // ASCII view of the 28x28 input: '#' = high value, '.' = low value
  let art: String = small
    .chunks(28)
    .map(|row| {
      row
        .iter()
        .map(|&x| {
          if x > 0.5 {
            '#'
          } else if x > 0.2 {
            '+'
          } else {
            '.'
          }
        })
        .collect::<String>()
        + "\n"
    })
    .collect();
  log!("{}", art);

  let input = Tensor::from_vec(small, (1, 784), device).map_err(|e| e.to_string())?;
  let output = model::model_forward(&weights, &input).map_err(|e| e.to_string())?;
  let values = output.to_vec2::<f32>().map_err(|e| e.to_string())?;
  log!("{:?}", values);
  let (digit, _prob) = values[0]
    .iter()
    .enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    .ok_or("empty output vector")?;
  log!("{:?}", digit);
  Ok(digit)
}

fn main() {
  mount_to_body(App);
}
