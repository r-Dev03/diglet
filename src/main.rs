mod model;
use std::cmp::max_by_key;
use std::usize;
use::std::cmp;

use js_sys::wasm_bindgen::JsCast;
use js_sys::Math;
use leptos::attr::Download;
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
          ctx.set_fill_style_str("black");
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
          context.set_stroke_style(&wasm_bindgen::JsValue::from_str("white"));
          context.set_line_width(36.0);
          context.set_line_cap("round");
          context.set_line_join("round");
          // context.set_line_width(4.0);
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
      if let Some(canvas) = canvas_ref.get() {
        let width = canvas.width();


        let image = context.get_image_data(0.0, 0.0, width as f64, width as f64);
        let image_data = image.unwrap().data();

        let mut greyscale: Vec<f32> = Vec::new();
        let (chunks, _rest) = image_data.as_chunks::<4>();


        for &[r, g, b, a] in chunks {
          let grey = f32::from(r) * 0.299 + f32::from(g) * 0.587 + f32::from(b) * 0.114; 
          let normalized = (grey / 255.0);
          greyscale.push(normalized); // Resolve polarity bug
        }

        match draw(greyscale, width as usize) {
          Ok(digit) => set_prediction.set(digit.to_string()),
          Err(e) => log!("inference failed: {e}")
        }

      }
    }

  }

    style="border: 1px solid black; width: 500px; height: 500px;"
      />

      <button on:click=move |_ev| {
        if let Some(context) = ctx.get() {
          context.clear_rect(0.0, 0.0, 500.0, 500.0);
          context.set_fill_style_str("black");
          context.fill_rect(0.0, 0.0, 500.0, 500.0);
          set_coordinates.set((0, 0));
        } 
      }>
      <span class="button_top">"Clear Canvas"</span>
      </button>
      <span>"Prediction: " {move || prediction.get()}</span>
  }
}

fn downsample(data: &[f32], src: usize, dst: usize) -> Vec<f32> {
  let mut out = vec![0.0f32; dst * dst];
  // index = oy * dst + ox
  for oy in 0..dst {

    let y0 = oy * src / dst;
    let y1   = (oy + 1) * src / dst;

    for ox in 0..dst {

      let x0 = ox * src / dst;
      let x1   = (ox + 1) * src / dst;

      let idx = oy * dst + ox;
      let mut total:f32 = 0.0;


      for y in y0..y1 {          
        for x in x0..x1 {      
          let src_idx = y * src + x;
          total += data[src_idx];
          
        }
      }

      let count = ((y1 - y0) * (x1 - x0)) as f32;
      let avg = total/&count;
      out[idx] = avg;

    }

  }
  out
}


fn draw(data: Vec<f32>, size: usize) -> Result<usize, String> {

  if data.len() != size * size {
    return Err(format!(
      "expected {} values for a {size}x{size} canvas, got {}",
      size * size,
      data.len()
    ));
  }

  let bytes = include_bytes!("../public/weights.safetensors");
  let weights = model::load_weights(bytes)
    .map_err(|e| format!("failed to load weights: {e}"))?;

  let device = &Device::Cpu;
  let small = downsample(&data, size, 28);

  // ASCII view of the 28x28 input: '#' = high value, '.' = low value
  let art: String = small.chunks(28)
    .map(|row| row.iter()
      .map(|&x| if x > 0.5 { '#' } else if x > 0.2 { '+' } else { '.' })
      .collect::<String>() + "\n")
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


fn bounding_box(data: &[f32], size: usize) -> Option<(usize, usize, usize, usize)> {
  const INK_THRESHOLD: f32 = 0.1;

  let mut x_min = usize::MAX;
  let mut y_min = usize::MAX;
  let mut x_max = 0;
  let mut y_max = 0;

  for y in 0..size {
    for x in 0..size {

      if data[y * size + x]  > INK_THRESHOLD {
        x_min = x_min.min(x);
        y_min = y_min.min(y);
        x_max = x_max.max(x);
        y_max = y_max.max(y);
      }

    }
  }

  if x_min > x_max {
    return None;
  }
  Some((x_min, y_min, x_max, y_max))
}


fn main() {
  mount_to_body(App);
}



