//! Llama 3 model configuration.
//!
//! Defines the hyperparameters for the Llama family of models, loaded from
//! the safetensors metadata or a separate config.json file.

use serde::Deserialize;

/// Configuration for a Llama-family transformer model.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size (e.g., 128256 for Llama 3).
    pub vocab_size: usize,

    /// Hidden dimension of the model (e.g., 4096 for 8B).
    pub hidden_size: usize,

    /// Intermediate dimension for the feed-forward network (e.g., 14336 for 8B).
    pub intermediate_size: usize,

    /// Number of transformer layers (e.g., 32 for 8B).
    pub num_hidden_layers: usize,

    /// Number of attention heads for queries (e.g., 32 for 8B).
    pub num_attention_heads: usize,

    /// Number of attention heads for keys and values (GQA).
    /// Fewer KV heads than query heads reduces KV cache size.
    /// (e.g., 8 for Llama 3 8B).
    pub num_key_value_heads: usize,

    /// Dimension of each attention head.
    /// Computed as hidden_size / num_attention_heads.
    #[serde(default)]
    pub head_dim: usize,

    /// RMSNorm epsilon (default 1e-5).
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    /// RoPE base frequency (default 10000.0 for Llama 3, 500000.0 for Llama 3.1+).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// RoPE scaling factor (default 1.0).
    #[serde(default = "default_rope_scale")]
    pub rope_scaling: f32,

    /// Maximum sequence length (context window).
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Whether to use traditional (interleaved) RoPE.
    #[serde(default)]
    pub rope_traditional: bool,

    /// Quantization bit-depth (0 means full precision).
    #[serde(default)]
    pub quantization_bits: u32,

    /// Quantization group size (e.g., 32 or 64).
    #[serde(default = "default_group_size")]
    pub quantization_group_size: i32,

    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_rope_scale() -> f32 {
    1.0
}
fn default_max_position_embeddings() -> usize {
    8192
}
fn default_group_size() -> i32 {
    64
}
fn default_model_type() -> String {
    "llama".to_string()
}

impl ModelConfig {
    /// Compute derived values after deserialization.
    pub fn resolve(&mut self) {
        if self.head_dim == 0 {
            self.head_dim = self.hidden_size / self.num_attention_heads;
        }
    }

    /// Number of query heads per KV head (GQA ratio).
    pub fn num_queries_per_kv(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Whether the model uses Grouped Query Attention.
    pub fn uses_gqa(&self) -> bool {
        self.num_key_value_heads < self.num_attention_heads
    }

    /// Whether the model weights are quantized.
    pub fn is_quantized(&self) -> bool {
        self.quantization_bits > 0
    }

    /// Estimated model memory in bytes (weights only, quantized if applicable).
    pub fn estimated_memory_bytes(&self) -> usize {
        let total_params = self.estimated_params();
        if self.is_quantized() {
            // quantized: bits per param
            total_params * (self.quantization_bits as usize) / 8
        } else {
            // float16: 2 bytes per param
            total_params * 2
        }
    }

    /// Rough estimate of total parameter count.
    pub fn estimated_params(&self) -> usize {
        // Embedding
        let embed = self.vocab_size * self.hidden_size;
        // Per-layer: attention (Q, K, V, O projections) + FFN (gate, up, down) + norms
        let attn = self.hidden_size * self.hidden_size // Q
            + self.hidden_size * (self.num_key_value_heads * self.head_dim) // K
            + self.hidden_size * (self.num_key_value_heads * self.head_dim) // V
            + self.hidden_size * self.hidden_size; // O
        let ffn = self.hidden_size * self.intermediate_size * 3; // gate + up + down
        let norms = self.hidden_size * 2; // attention_norm + ffn_norm
        let per_layer = attn + ffn + norms;
        // Output: final norm + lm_head
        let output = self.hidden_size + self.vocab_size * self.hidden_size;
        embed + self.num_hidden_layers * per_layer + output
    }

    /// Preset configuration for Llama 3 8B.
    pub fn llama3_8b() -> Self {
        let mut cfg = ModelConfig {
            vocab_size: 128256,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            rope_scaling: 1.0,
            max_position_embeddings: 8192,
            rope_traditional: false,
            quantization_bits: 0,
            quantization_group_size: 64,
            model_type: "llama".to_string(),
        };
        cfg.resolve();
        cfg
    }

    /// Preset configuration for Llama 3 70B.
    pub fn llama3_70b() -> Self {
        let mut cfg = ModelConfig {
            vocab_size: 128256,
            hidden_size: 8192,
            intermediate_size: 28672,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            rope_scaling: 1.0,
            max_position_embeddings: 8192,
            rope_traditional: false,
            quantization_bits: 0,
            quantization_group_size: 64,
            model_type: "llama".to_string(),
        };
        cfg.resolve();
        cfg
    }
}
