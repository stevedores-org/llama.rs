//! Rotary Position Embeddings (RoPE) for Llama models.
//!
//! RoPE injects positional information by rotating Query and Key vectors in
//! the complex plane. Llama uses the non-interleaved ("GPT-NeoX") variant.
//!
//! # Offset Management (Critical)
//!
//! - **Prefill phase**: positions are `0..N`, offset = 0.
//! - **Decode phase**: input is a single token at position N, offset = N.
//!
//! Passing `offset=0` during decoding is a common bug that causes hallucination,
//! because the model treats every generated token as position 0.

use crate::mlx::ops;
use crate::mlx::Array;

/// Configuration for RoPE embeddings.
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Number of dimensions to rotate (typically head_dim).
    pub dims: i32,

    /// Whether to use traditional (interleaved) RoPE layout.
    /// Llama 3 uses non-traditional (GPT-NeoX style).
    pub traditional: bool,

    /// Base frequency. Default 10000.0 for Llama 2, 500000.0 for Llama 3.1+.
    pub base: f32,

    /// Scaling factor for extended context. Default 1.0.
    pub scale: f32,
}

impl Default for RopeConfig {
    fn default() -> Self {
        RopeConfig {
            dims: 128,
            traditional: false,
            base: 500000.0,
            scale: 1.0,
        }
    }
}

/// Apply Rotary Position Embeddings to a tensor.
///
/// Uses the optimized `mlx::fast::rope` Metal kernel for high performance.
///
/// # Arguments
/// - `x`: Input tensor of shape `[batch, seq_len, n_heads, head_dim]`.
/// - `config`: RoPE hyperparameters.
/// - `offset`: Position offset. **Must be set to the current sequence position
///   during the decode phase.** Set to 0 during prefill.
///
/// # Returns
/// Tensor with the same shape, with rotational embeddings applied.
pub fn apply_rope(x: &Array, config: &RopeConfig, offset: i32) -> Array {
    ops::fast_rope(x, config.dims, config.traditional, config.base, config.scale, offset)
}
