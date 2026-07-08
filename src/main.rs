mod model;
use js_sys::wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement,};
use leptos::html::{Canvas, HtmlElement};
use leptos::{prelude::*, svg::view, html};
use candle_core::{Device, Tensor, DType, Result as CandleResult, Shape};
use leptos::logging::log;
use leptos::prelude::*;



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
          let display_size = 500.0; 

          // set physical canvas internal resolution size 
          canvas.set_width((display_size * dpr) as u32);
          canvas.set_height((display_size * dpr) as u32);

          // scale drawing context so standard layout coords map perfectly to high-res pixels
          ctx.scale(dpr, dpr).unwrap();
          ctx.set_fill_style_str("white");
          ctx.fill_rect(0.0, 0.0, display_size, display_size);
          set_ctx.set(Some(ctx));  
        }
      }
    }
  });

  view! {
  <canvas 
  node_ref=canvas_ref
  on:mousedown=move |ev| { // mouse down -> enable drawing
    if let Some(canvas) = canvas_ref.get() {
      let rect = canvas.get_bounding_client_rect();
      let x = ev.client_x() as f64 - rect.left();
      let y = ev.client_y() as f64 - rect.top();
      set_coordinates.set((x as i32, y as i32));
      set_is_drawing.set(true);
    }
  }

  on:mousemove=move |ev| { // mouse move -> start drawing
    if let Some(canvas) = canvas_ref.get() {
      let rect = canvas.get_bounding_client_rect();
      let x = ev.client_x() as f64 - rect.left();
      let y = ev.client_y() as f64 - rect.top();

      if let Some(context) = ctx.get() { 
        if is_drawing.get() == true {
          context.begin_path();
          context.set_stroke_style(&wasm_bindgen::JsValue::from_str("black"));
          context.set_line_width(4.0);
          context.move_to(coordinates.get().0 as f64, coordinates.get().1 as f64);
          context.line_to(x as f64, y as f64);
          context.stroke();
        }

        set_coordinates.set((x as i32, y as i32));
      }

    }
  }

  on:mouseup=move |ev| { // mouse up -> stop drawing & and retrieve data
    set_is_drawing.set(false); 
    if let Some(context) = ctx.get() {
      let window = web_sys::window().unwrap();
      let dpr = window.device_pixel_ratio();

      let display_size = 500.0;
      let scaled_size = display_size * dpr;


      let image = context.get_image_data(0.0, 0.0, scaled_size, scaled_size);
      let image_data = image.unwrap().data();

      let mut greyscale: Vec<f32> = Vec::new();
      let (chunks, _rest) = image_data.as_chunks::<4>();


      for &[r, g, b, a] in chunks {
        let grey = f32::from(r) * 0.299 + f32::from(g) * 0.587 + f32::from(b) * 0.114;
        let normalized = (grey / 255.0);
        greyscale.push(normalized);
      }

      match draw(greyscale) {
        Ok(digit) => set_prediction.set(digit.to_string()),
        Err(e) => log!("inference failed: {e}")
      }

    }

  }

    style="border: 1px solid black; width: 500px; height: 500px;"
      />

      <button on:click=move |_ev| {
        if let Some(context) = ctx.get() {
          context.clear_rect(0.0, 0.0, 500.0, 500.0);
          context.set_fill_style_str("white");
          context.fill_rect(0.0, 0.0, 500.0, 500.0);
          set_coordinates.set((0, 0));
        } 
      }>
      <span class="button_top">"Clear Canvas"</span>
      </button>
      <span>"Prediction: " {move || prediction.get()}</span>
  }
}

fn main() {
  mount_to_body(App);
}



fn draw(data: Vec<f32>) -> Result<usize, String> {
  let weights = model::load_weights("/model/weights.safetensors")
    .map_err(|e| format!("failed to load weights: {e}"))?;
  let device = &Device::Cpu;
  let input_4d = Tensor::from_vec(data, (1, 1, 500, 500), device)
    .map_err(|e| e.to_string())?;
  let resized_4d = input_4d.interpolate2d(28, 28).map_err(|e| e.to_string())?;
  let input = resized_4d.flatten_all().unwrap().reshape((1,784)).map_err(|e| e.to_string())?;
  let output = model::model_forward(&weights, &input).map_err(|e| e.to_string())?;
  let values = output.to_vec2::<f32>().map_err(|e| e.to_string())?;
  let (digit, _prob) = values[0]
    .iter()
    .enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    .ok_or("empty output vector")?;
  Ok(digit)
}
