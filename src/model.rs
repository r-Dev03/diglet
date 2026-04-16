use std::io;
use std::fs;
use::std::error::Error;
use::std::collections;
use candle_nn;
use candle_core::Device;
use::candle_core::Tensor;
use candle_core::safetensors;

pub struct ModelWeights {
  pub w1: Tensor,
  pub b1: Tensor,
  pub w2: Tensor,
  pub b2: Tensor,
}

pub fn load_weights(path: &str) -> Result<ModelWeights, Box<dyn Error>> {
  let device = Device::Cpu;
  let mut tensors = safetensors::load(path, &device)?;

  let w1 = tensors.remove("w1").ok_or("w1 not found")?;
  let b1 = tensors.remove("b1").ok_or("b1 not found")?;
  let w2 = tensors.remove("w2").ok_or("w2 not found")?;
  let b2 = tensors.remove("b2").ok_or("b2 not found")?;

  let weights = ModelWeights { w1, b1, w2, b2 };

  Ok(weights)
}


pub fn model_forward(weights: &ModelWeights, inputs: &Tensor) -> Result<Tensor, Box<dyn Error>> {
  let hidden1 = inputs.matmul(&weights.w1)?.broadcast_add(&weights.b1)?.relu()?;
  let hidden2 = hidden1.matmul(&weights.w2)?.broadcast_add(&weights.b2)?;
  let output = candle_nn::ops::softmax(&hidden2, 1)?;

  Ok(output)
}



// reference
// def model_forward(weights: ModelWeights, inputs: Array) -> Array:
//     hidden1 = jax.nn.relu(jnp.matmul(inputs, weights.w1) + weights.b1)
//     hidden2 = jnp.matmul(hidden1, weights.w2) + weights.b2
//     # return hidden2
//     return jax.nn.softmax(hidden2)
//

