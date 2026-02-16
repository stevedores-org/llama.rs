//! Llama 3 model architecture.
//!
//! This module implements the complete Llama 3 transformer architecture
//! optimized for Apple Silicon inference via MLX:
//!
//! - [`config`]: Model hyperparameters and presets.
//! - [`rope`]: Rotary Position Embeddings with offset management.
//! - [`norm`]: RMSNorm normalization.
//! - [`linear`]: Full-precision and quantized linear layers.
//! - [`embedding`]: Token embedding lookup.
//! - [`attention`]: Grouped Query Attention with FlashAttention.
//! - [`transformer`]: Full model composition.

pub mod attention;
pub mod config;
pub mod embedding;
pub mod linear;
pub mod norm;
pub mod rope;
pub mod transformer;

pub use config::ModelConfig;
pub use transformer::LlamaModel;
