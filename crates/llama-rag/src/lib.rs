//! # llama-rag
//!
//! Adapter bridging `llama-engine` to `oxidizedRAG` (graphrag-core).
//!
//! Provides:
//! - Synchronous adapters:
//!   - [`LlamaEmbedder`]: Implements graphrag-core's `Embedder` trait using `LlamaEngine::embed()`
//!   - [`RagPipeline`]: End-to-end query → retrieve → generate pipeline
//!   - [`RagPromptBuilder`]: Builds retrieval-augmented prompts with inline citations
//! - Async adapters:
//!   - [`LlamaAsyncEmbedder`]: Implements `AsyncEmbedder` trait with concurrent support
//!   - [`LlamaAsyncLanguageModel`]: Implements `AsyncLanguageModel` trait for generation

mod async_embedder;
mod config;
mod embedder;
mod error;
mod language_model;
mod pipeline;
mod prompt;

pub use async_embedder::LlamaAsyncEmbedder;
pub use config::RagConfig;
pub use embedder::LlamaEmbedder;
pub use error::{RagError, Result};
pub use language_model::LlamaAsyncLanguageModel;
pub use pipeline::{RagPipeline, RagResult};
pub use prompt::{Citation, RagPromptBuilder};

// Re-export for convenience
pub use llama_engine::{LlamaEngine, Session};
