//! Error types for the llama-rs crate.

use thiserror::Error;

/// Top-level error type for llama-rs operations.
#[derive(Error, Debug)]
pub enum LlamaError {
    #[error("MLX error: {0}")]
    Mlx(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Weight loading error: {0}")]
    WeightLoad(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Out of memory: {0}")]
    OutOfMemory(String),
}

pub type Result<T> = std::result::Result<T, LlamaError>;
