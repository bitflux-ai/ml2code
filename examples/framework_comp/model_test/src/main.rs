use model::TinyModel;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::Instant;
use clap::Parser;
use tract_onnx::prelude::*;
use ort::{session::builder::SessionBuilder, value::Value, session::SessionInputValue};
use candle_core::{Device, Tensor};
use candle_onnx;

#[derive(Parser)]
#[command(name = "model_test")]
#[command(about = "Compare native Rust model vs ONNX tract runtime")]
struct Args {
    /// Input binary file (128 bytes, 32 f32 values). If not provided, auto-generates input.
    #[arg(long)]
    input: Option<String>,
    
    /// Output binary file (16 bytes, 4 f32 values). If not provided, no output file is written.
    #[arg(long)]
    output: Option<String>,
    
    /// Number of inference runs
    #[arg(long, default_value = "1")]
    count: i32,
    
    /// Path to ONNX model file
    #[arg(long)]
    model: Option<String>,
}

fn format_duration(duration_ns: u128) -> String {
    if duration_ns >= 1_000_000_000 {
        format!("{:.2} s", duration_ns as f64 / 1_000_000_000.0)
    } else if duration_ns >= 1_000_000 {
        format!("{:.2} ms", duration_ns as f64 / 1_000_000.0)
    } else if duration_ns >= 1_000 {
        format!("{:.2} us", duration_ns as f64 / 1_000.0)
    } else {
        format!("{} ns", duration_ns)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args = Args::parse();

  // Create an input buffer of 32 f32s
  let mut input = [0.0; 32];

  // Create an output buffer of 4 f32s
  let mut output = [0.0; 4];

  // Read inputs from file or auto-generate
  if let Some(input_path) = &args.input {
    let mut f = File::open(input_path)?;
    let mut input_bytes = Vec::new();
    f.read_to_end(&mut input_bytes)?;
    if input_bytes.len() != 128 {
      return Err("Invalid input file size".into());
    }

    // Map the input_bytes into input
    for i in 0..32 {
      input[i] = f32::from_le_bytes([input_bytes[i*4+0], input_bytes[i*4+1], input_bytes[i*4+2], input_bytes[i*4+3]]);
    }
  } else {
    // Auto-generate input values (simple pattern for testing)
    for i in 0..32 {
      input[i] = (i as f32 + 1.0) / 32.0;
    }
  }

  // Run native Rust model
  let mut net = TinyModel::new();
  let start = Instant::now();
  for _ in 0..args.count {
    net.run(&input, &mut output);
  }
  let rust_duration = start.elapsed().as_nanos();

  // Run ONNX models if provided
  let mut tract_duration: Option<u128> = None;
  let mut ort_duration: Option<u128> = None;
  let mut candle_duration: Option<u128> = None;
  if let Some(onnx_path) = &args.model {
    // Extract input shape from ONNX model for generating inputs_onnx
    let onnx_model = candle_onnx::read_file(onnx_path)?;
    let graph = onnx_model.graph.as_ref().ok_or("No graph in ONNX model")?;
    let input_info = graph.input.first().ok_or("No input found in ONNX model")?;
    
    // Extract shape from input tensor type
    let mut onnx_input_shape = Vec::new();
    let mut onnx_input_size = 1usize;
    if let Some(type_proto) = &input_info.r#type {
      if let Some(value) = &type_proto.value {
        match value {
          candle_onnx::onnx::type_proto::Value::TensorType(tensor_type) => {
            if let Some(shape) = &tensor_type.shape {
              for dim in &shape.dim {
                if let Some(dim_value) = &dim.value {
                  match dim_value {
                    candle_onnx::onnx::tensor_shape_proto::dimension::Value::DimValue(v) => {
                      if *v > 0 {
                        let dim_size = (*v) as usize;
                        onnx_input_shape.push(dim_size);
                        onnx_input_size *= dim_size;
                      }
                    }
                    _ => {} // Skip symbolic dimensions
                  }
                }
              }
            }
          }
          _ => {} // Skip non-tensor types
        }
      }
    }
    
    // Generate inputs_onnx based on ONNX model's expected input size
    let inputs_onnx: Vec<f32> = (0..onnx_input_size)
      .map(|i| (i as f32 + 1.0) / onnx_input_size as f32)
      .collect();
    // Tract runtime (single-threaded by default)
    match (|| -> Result<u128, Box<dyn std::error::Error>> {
      let model = tract_onnx::onnx()
        .model_for_path(onnx_path)?
        .into_optimized()?
        .into_runnable()?;
      
      // Convert inputs_onnx to tract tensor
      let input_tensor = if onnx_input_shape.len() > 2 {
        tract_ndarray::ArrayD::from_shape_vec(onnx_input_shape.clone(), inputs_onnx.clone())?
          .into_tensor()
      } else {
        tract_ndarray::Array2::from_shape_vec((1, onnx_input_size), inputs_onnx.clone())?
          .into_dyn()
          .into_tensor()
      };
      
      let start = Instant::now();
      for _ in 0..args.count {
        let _result = model.run(tvec!(input_tensor.clone().into()))?;
      }
      Ok(start.elapsed().as_nanos())
    })() {
      Ok(duration) => tract_duration = Some(duration),
      Err(e) => eprintln!("Tract failed: {}", e),
    }

    // ORT runtime
    match (|| -> Result<u128, Box<dyn std::error::Error>> {
      let mut session = SessionBuilder::new()?
        .with_intra_threads(1)?  // Use single thread for intra-op parallelism
        .with_inter_threads(1)?  // Use single thread for inter-op parallelism
        .commit_from_file(onnx_path)?;
      
      let start = Instant::now();
      for _ in 0..args.count {
        let input_array = if onnx_input_shape.len() > 2 {
          Value::from_array(
            ndarray::ArrayD::from_shape_vec(onnx_input_shape.clone(), inputs_onnx.clone())?
          )?
        } else {
          Value::from_array(
            ndarray::Array2::from_shape_vec((1, onnx_input_size), inputs_onnx.clone())?
          )?
        };
        let _outputs = session.run([SessionInputValue::from(input_array)])?;
      }
      Ok(start.elapsed().as_nanos())
    })() {
      Ok(duration) => ort_duration = Some(duration),
      Err(e) => eprintln!("ORT failed: {}", e),
    }

    // Candle runtime (single-threaded by default)
    match (|| -> Result<u128, Box<dyn std::error::Error>> {
      let device = Device::Cpu;
      
      let start = Instant::now();
      for _ in 0..args.count {
        let input_tensor = if onnx_input_shape.len() > 2 {
          Tensor::from_vec(inputs_onnx.clone(), onnx_input_shape.as_slice(), &device)?
        } else {
          Tensor::from_vec(inputs_onnx.clone(), (1, onnx_input_size), &device)?
        };
        let mut inputs = std::collections::HashMap::new();
        if let Some(input_node) = graph.input.first() {
          inputs.insert(input_node.name.clone(), input_tensor);
        }
        let _outputs = candle_onnx::simple_eval(&onnx_model, inputs)?;
      }
      Ok(start.elapsed().as_nanos())
    })() {
      Ok(duration) => candle_duration = Some(duration),
      Err(e) => eprintln!("Candle failed: {}", e),
    }
  }

  // Print timing results
  println!("rust_src: {}", format_duration(rust_duration));
  if args.model.is_some() {
    if let Some(duration) = tract_duration {
      println!("tract: {}", format_duration(duration));
    }
    if let Some(duration) = ort_duration {
      println!("ort: {}", format_duration(duration));
    }
    if let Some(duration) = candle_duration {
      println!("candle: {}", format_duration(duration));
    }
  }

  // Write output file if specified
  if let Some(output_path) = &args.output {
    let mut output_file = OpenOptions::new()
      .write(true)
      .create(true)
      .truncate(true)
      .open(output_path)?;
    for o in output.iter() {
        output_file.write_all(&o.to_le_bytes())?;
    }
  }

  Ok(())
}
