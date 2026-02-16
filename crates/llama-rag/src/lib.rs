//! # llama-rag
//!
//! Adapter bridging `llama-engine` to `oxidizedRAG` (graphrag-core).
//!
//! Provides:
//! - [`LlamaEmbedder`]: Implements graphrag-core's `Embedder` trait using `LlamaEngine::embed()`
//! - [`RagPromptBuilder`]: Builds retrieval-augmented prompts with inline citations
//! - [`RagPipeline`]: End-to-end query → retrieve → generate pipeline

mod embedder;
mod pipeline;
mod prompt;

pub use embedder::LlamaEmbedder;
pub use pipeline::{RagPipeline, RagResult};
pub use prompt::{Citation, RagPromptBuilder};
