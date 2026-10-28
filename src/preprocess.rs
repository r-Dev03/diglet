/// turns a raw canvas into a 28x28 MNIST-style input, or None if nothing is drawn.
pub fn preprocess(canvas: &[f32], canvas_size: usize) -> Option<Vec<f32>> {
  const DIGIT_SIZE: usize = 20; // MNIST digits fit in a 20x20 box
  const IMAGE_SIZE: usize = 28; // centered inside a 28x28 image

  let bbox = bounding_box(canvas, canvas_size)?;
  let (square, square_size) = crop_square(canvas, canvas_size, bbox, DIGIT_SIZE);
  let digit = downsample(&square, square_size, DIGIT_SIZE);
  let image = center_by_mass(&digit, DIGIT_SIZE, IMAGE_SIZE);

  Some(image)
}

/// smallest box that contains the digit, returning the edges as a tuple, or None if the canvas is blank
fn bounding_box(canvas: &[f32], canvas_size: usize) -> Option<(usize, usize, usize, usize)> {
  const INK_THRESHOLD: f32 = 0.1;

  let mut col_min = usize::MAX;
  let mut row_min = usize::MAX;
  let mut col_max = 0;
  let mut row_max = 0;

  for row in 0..canvas_size {
    for col in 0..canvas_size {
      if canvas[row * canvas_size + col] > INK_THRESHOLD {
        col_min = col_min.min(col);
        row_min = row_min.min(row);
        col_max = col_max.max(col);
        row_max = row_max.max(row);
      }
    }
  }

  if col_min > col_max {
    return None; // no ink found
  }

  Some((col_min, row_min, col_max, row_max))
}

/// cuts a square around the bounding box, with the box centered inside it
fn crop_square(
  canvas: &[f32],
  canvas_size: usize,
  bbox: (usize, usize, usize, usize),
  min_size: usize,
) -> (Vec<f32>, usize) {
  let (col_min, row_min, col_max, row_max) = bbox;

  let box_width = col_max - col_min + 1;
  let box_height = row_max - row_min + 1;

  // use the longer side, but never smaller than min_size
  let square_size = box_width.max(box_height).max(min_size);

  // move the top-left corner back by half the leftover space, so the digit ends up centered
  // signed, because near the canvas edge it can be negative
  let start_col = col_min as isize - ((square_size - box_width) / 2) as isize;
  let start_row = row_min as isize - ((square_size - box_height) / 2) as isize;

  let mut square = vec![0.0f32; square_size * square_size];

  for square_row in 0..square_size {
    for square_col in 0..square_size {
      let canvas_row = start_row + square_row as isize;
      let canvas_col = start_col + square_col as isize;

      let on_canvas = canvas_row >= 0
        && canvas_col >= 0
        && canvas_row < canvas_size as isize
        && canvas_col < canvas_size as isize;

      // if it's off the canvas, it doesn't exist (0.0)
      if on_canvas {
        square[square_row * square_size + square_col] =
          canvas[canvas_row as usize * canvas_size + canvas_col as usize];
      }
    }
  }

  (square, square_size)
}

/// shrinks the square to dst_size x dst_size by averaging each block of pixels
fn downsample(src: &[f32], src_size: usize, dst_size: usize) -> Vec<f32> {
  let mut dst = vec![0.0f32; dst_size * dst_size];

  for dst_row in 0..dst_size {
    // which source rows this output row covers (end is exclusive)
    let row_start = dst_row * src_size / dst_size;
    let row_end = (dst_row + 1) * src_size / dst_size;

    for dst_col in 0..dst_size {
      // which source columns this output column covers
      let col_start = dst_col * src_size / dst_size;
      let col_end = (dst_col + 1) * src_size / dst_size;

      let mut total: f32 = 0.0;
      for src_row in row_start..row_end {
        for src_col in col_start..col_end {
          total += src[src_row * src_size + src_col];
        }
      }

      let pixel_count = ((row_end - row_start) * (col_end - col_start)) as f32;
      dst[dst_row * dst_size + dst_col] = total / pixel_count;
    }
  }

  dst
}

/// places the digit into a blank 28x28 image so its center of mass lands at the center, like MNIST
fn center_by_mass(digit: &[f32], digit_size: usize, image_size: usize) -> Vec<f32> {
  // center of mass: brightness-weighted average row and column.
  let mut total_mass: f32 = 0.0;
  let mut weighted_col_sum: f32 = 0.0;
  let mut weighted_row_sum: f32 = 0.0;

  for digit_row in 0..digit_size {
    for digit_col in 0..digit_size {
      let brightness = digit[digit_row * digit_size + digit_col];
      total_mass += brightness;
      weighted_col_sum += brightness * digit_col as f32;
      weighted_row_sum += brightness * digit_row as f32;
    }
  }

  let mass_col = weighted_col_sum / total_mass;
  let mass_row = weighted_row_sum / total_mass;

  // shift so the center of mass lands on the image center, kept inside the frame.
  let image_center = (image_size - 1) as f32 / 2.0;
  let max_offset = (image_size - digit_size) as i32;
  let offset_col = ((image_center - mass_col).round() as i32).clamp(0, max_offset) as usize;
  let offset_row = ((image_center - mass_row).round() as i32).clamp(0, max_offset) as usize;

  let mut image = vec![0.0f32; image_size * image_size];

  for digit_row in 0..digit_size {
    for digit_col in 0..digit_size {
      let image_row = digit_row + offset_row;
      let image_col = digit_col + offset_col;
      image[image_row * image_size + image_col] = digit[digit_row * digit_size + digit_col];
    }
  }

  image
}
