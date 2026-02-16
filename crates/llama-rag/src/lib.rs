//! # llama-rag
//!
//! Integration point for Retrieval-Augmented Generation in llama.rs.
//! Provides traits for connecting to external vector databases and
//! performing context injection.

use llama_engine::LlamaEngine;

/// Error type for RAG operations.
#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error("Retrieval failed: {0}")]
    Retrieval(String),
    #[error("Engine error: {0}")]
    Engine(#[from] llama_engine::LlamaError),
}

pub type Result<T> = std::result::Result<T, RagError>;

/// Represents a retrieved document chunk.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub score: f32,
}

/// Core trait for retrieval.
///
/// Implementations can wrap `oxidizedRAG` or other vector stores.
pub trait Retriever: Send + Sync {
    /// Retrieve relevant chunks for a query.
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<Chunk>>;
}

/// A thin wrapper that combines an engine with a retriever.
pub struct RagWorkflow<E: LlamaEngine, R: Retriever> {
    pub engine: E,
    pub retriever: R,
}

impl<E: LlamaEngine, R: Retriever> RagWorkflow<E, R> {
    pub fn new(engine: E, retriever: R) -> Self {
        Self { engine, retriever }
    }

    /// Basic RAG flow: Retrieve -> Augment -> Generate (simplified)
    pub fn query(&self, query: &str) -> Result<String> {
        let chunks = self.retriever.retrieve(query, 3)?;

        let context = chunks.iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        let augmented_prompt = format!(
            "Context:\n{}\n\nQuestion: {}\n\nAnswer:",
            context, query
        );

        // For Milestone A demo, we just return the augmented prompt length
        // In a real impl, this would call engine.prefill and engine.decode
        let tokens = self.engine.tokenize(&augmented_prompt)?;
        Ok(format!("Retrieved {} chunks, augmented prompt is {} tokens", chunks.len(), tokens.len()))
    }
}
