use std::io;
use std::fs;
use::std::error::Error;
use::std::collections;
use candle_core::Device;
use candle_core::safetensors;

pub struct ModelWeights {
  pub w1: Vec<f32>,
  pub b1: Vec<f32>,
  pub w2: Vec<f32>,
  pub b2: Vec<f32>,
}

pub fn load_weights(path: &str) -> Result<ModelWeights, Box<dyn Error>> {
  let device = Device::Cpu;
  let tensors = safetensors::load(path, &device)?;

  let w1 = tensors.get("w1");
  let b1 = tensors.get("b1");
  let w2 = tensors.get("w2");
  let b2 = tensors.get("b2");

  let weights = ModelWeights {

    w1: w1,
    b1: b1,
    w2: w2,
    b2: b2,

  };


  Ok(weights)
}

