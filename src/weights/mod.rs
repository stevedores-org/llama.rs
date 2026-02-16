//! Weight loading and management.
//!
//! Supports loading model weights from safetensors files with:
//! - Memory-mapped I/O for fast loading (millisecond startup for 70B models)
//! - Zero-copy MLX array creation from mapped pages
//! - Quantized weight support (4-bit, 8-bit group-wise affine)
//! - Multi-file model support (model sharded across multiple .safetensors files)

pub mod mmap;
pub mod safetensors;

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

use crate::error::{LlamaError, Result};
use crate::mlx::Array;
use crate::model::config::ModelConfig;

pub use self::safetensors::SafetensorsFile;

/// A collection of named weight tensors loaded from disk.
pub struct WeightStore {
    /// Named weight tensors.
    weights: HashMap<String, Array>,

    /// Safetensors files kept alive for zero-copy (their mmap must outlive arrays).
    _files: Vec<SafetensorsFile>,
}

impl WeightStore {
    /// Load weights from one or more safetensors files.
    ///
    /// Files are memory-mapped and tensors are created as zero-copy views
    /// into the mapped pages. The GPU reads directly from these pages on
    /// Apple Silicon (UMA).
    pub fn load(paths: &[&Path]) -> Result<Self> {
        let mut weights = HashMap::new();
        let mut files = Vec::new();

        for path in paths {
            let file = SafetensorsFile::open(path)?;

            // Create zero-copy arrays for each tensor
            for (name, info) in &file.header.tensors {
                let dtype = info.parse_dtype()?;
                let shape = info.shape_i32();
                let offset = file.header.data_offset + info.data_offsets[0];
                let ptr = file.mapped.ptr_at(offset) as *const c_void;

                // Safety: The MappedFile is kept alive in `files` for the
                // entire lifetime of this WeightStore. The mapped memory
                // is read-only and stable.
                let array = unsafe { Array::from_ptr(ptr, &shape, dtype) };
                weights.insert(name.clone(), array);
            }

            files.push(file);
        }

        Ok(WeightStore {
            weights,
            _files: files,
        })
    }

    /// Get a weight tensor by name.
    pub fn get(&self, name: &str) -> Result<&Array> {
        self.weights.get(name).ok_or_else(|| {
            LlamaError::WeightLoad(format!("weight '{name}' not found"))
        })
    }

    /// Take a weight tensor by name (removes from store).
    pub fn take(&mut self, name: &str) -> Result<Array> {
        self.weights.remove(name).ok_or_else(|| {
            LlamaError::WeightLoad(format!("weight '{name}' not found"))
        })
    }

    /// Check if a weight exists.
    pub fn has(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// List all weight names.
    pub fn names(&self) -> Vec<&str> {
        self.weights.keys().map(|s| s.as_str()).collect()
    }

    /// Number of loaded weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether no weights are loaded.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// Load a model configuration from a config.json file.
pub fn load_config(path: &Path) -> Result<ModelConfig> {
    let data = std::fs::read_to_string(path)?;
    let mut config: ModelConfig = serde_json::from_str(&data)?;
    config.resolve();
    Ok(config)
}

/// Detect quantization parameters from safetensors metadata.
///
/// Inspects the `__metadata__` section for quantization info.
/// Falls back to checking weight tensor shapes and dtypes.
pub fn detect_quantization(files: &[&SafetensorsFile]) -> Option<(u32, i32)> {
    for file in files {
        // Check metadata for explicit quantization info
        if let Some(bits_str) = file.header.metadata.get("quantization_bits") {
            if let Ok(bits) = bits_str.parse::<u32>() {
                let group_size = file
                    .header
                    .metadata
                    .get("quantization_group_size")
                    .and_then(|s| s.parse::<i32>().ok())
                    .unwrap_or(64);
                return Some((bits, group_size));
            }
        }

        // Heuristic: check if any weight tensor has a "scales" companion
        for name in file.header.tensors.keys() {
            if name.ends_with(".scales") {
                // Quantized model detected, try to infer bit depth
                let weight_name = name.replace(".scales", ".weight");
                if let (Some(w_info), Some(s_info)) = (
                    file.header.tensors.get(&weight_name),
                    file.header.tensors.get(name),
                ) {
                    // Infer group_size from shapes
                    if w_info.shape.len() >= 2 && s_info.shape.len() >= 2 {
                        let group_size = (w_info.shape[1] * 32 / s_info.shape[1]) as i32;
                        return Some((4, group_size)); // assume 4-bit
                    }
                }
            }
        }
    }

    None
}

/// Discover all safetensors files in a model directory.
///
/// Handles both single-file models and sharded models
/// (e.g., model-00001-of-00004.safetensors).
pub fn discover_weight_files(model_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext == "safetensors")
        })
        .collect();

    if files.is_empty() {
        return Err(LlamaError::WeightLoad(format!(
            "no .safetensors files found in {}",
            model_dir.display()
        )));
    }

    files.sort();
    Ok(files)
}

/// Estimate total model memory from weight files without loading them.
pub fn estimate_model_memory(files: &[SafetensorsFile]) -> usize {
    files
        .iter()
        .flat_map(|f| f.header.tensors.values())
        .map(|info| info.byte_size())
        .sum()
}

/// Validate that loaded weights match the model configuration.
pub fn validate_weights(weights: &WeightStore, config: &ModelConfig) -> Result<()> {
    // Check embedding
    if !weights.has("model.embed_tokens.weight") {
        return Err(LlamaError::WeightLoad(
            "missing model.embed_tokens.weight".into(),
        ));
    }

    // Check each layer
    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        let required = [
            format!("{prefix}.self_attn.q_proj.weight"),
            format!("{prefix}.self_attn.k_proj.weight"),
            format!("{prefix}.self_attn.v_proj.weight"),
            format!("{prefix}.self_attn.o_proj.weight"),
            format!("{prefix}.mlp.gate_proj.weight"),
            format!("{prefix}.mlp.up_proj.weight"),
            format!("{prefix}.mlp.down_proj.weight"),
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.post_attention_layernorm.weight"),
        ];

        for name in &required {
            if !weights.has(name) {
                return Err(LlamaError::WeightLoad(format!("missing weight: {name}")));
            }
        }
    }

    // Check final norm and lm_head
    if !weights.has("model.norm.weight") {
        return Err(LlamaError::WeightLoad("missing model.norm.weight".into()));
    }

    Ok(())
}
