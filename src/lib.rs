//! # llama-rs
//!
//! High-performance LLM inference on Apple Silicon using Rust and MLX.
//!
//! This crate provides a complete pipeline for running Llama-family models
//! locally, optimized for Apple Silicon's Unified Memory Architecture.
//!
//! ## Architecture
//!
//! - **MLX Bindings** ([`mlx`]): Safe Rust wrappers around the MLX framework,
//!   providing zero-copy array operations on Apple Silicon's unified memory.
//!
//! - **Model** ([`model`]): Llama 3 transformer architecture with Grouped Query
//!   Attention, RoPE embeddings, SwiGLU FFN, and RMSNorm.
//!
//! - **KV Cache** ([`cache`]): Pre-allocated key-value cache for O(1) updates
//!   during autoregressive generation (no allocation in the decode loop).
//!
//! - **Weights** ([`weights`]): Memory-mapped safetensors loading with zero-copy
//!   MLX array creation and quantized weight support (4-bit/8-bit).
//!
//! - **Sampling** ([`sampling`]): Temperature scaling, Top-K, Top-P (nucleus),
//!   and greedy sampling — all implemented on GPU via MLX primitives.
//!
//! - **Engine** ([`engine`]): Two-phase inference (prefill + decode) with
//!   actor-based concurrency for non-blocking UI integration.
//!
//! - **Tokenizer** ([`tokenizer`]): BPE tokenizer with Llama 3 chat template
//!   formatting.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use llama::session::{Session, SessionConfig};
//!
//! let config = SessionConfig {
//!     model_path: "/path/to/model".into(),
//!     tokenizer_path: "/path/to/tokenizer.json".into(),
//!     max_seq_len: 8192,
//!     ..Default::default()
//! };
//!
//! let mut session = Session::new(config).expect("failed to load model");
//! session.set_system_prompt("You are a helpful assistant.");
//! session.send_message("Hello!").expect("failed to send message");
//! ```

pub mod cache;
pub mod engine;
pub mod error;
pub mod mlx;
pub mod model;
pub mod sampling;
pub mod session;
pub mod tokenizer;
pub mod weights;

pub use error::{LlamaError, Result};
pub use model::config::ModelConfig;
pub use model::LlamaModel;
pub use session::Session;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
