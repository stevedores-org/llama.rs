//! Error types for llama-rag adapter.

use llama_engine::LlamaError;

/// Result type for llama-rag operations.
pub type Result<T> = std::result::Result<T, RagError>;

/// Errors that can occur in llama-rag operations.
#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("generation error: {0}")]
    Generation(String),

    #[error("tokenization error: {0}")]
    Tokenization(String),

    #[error("model unavailable: {0}")]
    ModelUnavailable(String),

    #[error("engine error: {0}")]
    Engine(#[from] LlamaError),
}

impl From<RagError> for graphrag_core::core::GraphRAGError {
    fn from(err: RagError) -> Self {
        match err {
            RagError::Embedding(msg) => {
                graphrag_core::core::GraphRAGError::Embedding { message: msg }
            }
            RagError::Generation(msg) => {
                graphrag_core::core::GraphRAGError::Generation { message: msg }
            }
            RagError::Tokenization(msg) => {
                graphrag_core::core::GraphRAGError::TextProcessing { message: msg }
            }
            RagError::ModelUnavailable(msg) => {
                graphrag_core::core::GraphRAGError::Config { message: msg }
            }
            RagError::Engine(e) => graphrag_core::core::GraphRAGError::Generation {
                message: e.to_string(),
            },
        }
    }
}
