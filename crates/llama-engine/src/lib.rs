//! # llama-engine
//!
//! The "narrow waist" of the llama.rs stack. Defines the core [`LlamaEngine`] trait
//! and associated types that all other crates depend on. Implementations can swap
//! CPU/Metal/FFI backends without changing application code.

pub type Result<T> = std::result::Result<T, LlamaError>;

/// Top-level error type for all engine operations.
#[derive(Debug, thiserror::Error)]
pub enum LlamaError {
    #[error("Model loading failed: {0}")]
    ModelLoad(String),
    #[error("Tokenization failed: {0}")]
    Tokenization(String),
    #[error("Inference failed: {0}")]
    Inference(String),
}

/// Specification for loading a model.
pub struct ModelSpec {
    pub path: String,
    pub context_size: usize,
}

/// Opaque handle to a loaded model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelHandle {
    pub id: u64,
}

/// Represents an active inference session with its own KV cache state.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Session {
    pub id: u64,
}

/// Result of the prefill phase (prompt processing).
pub struct PrefillResult;

/// Streaming token output from the decode phase.
pub struct TokenStream;

/// The core engine trait — everything else plugs into this.
///
/// oxidizedRAG and oxidizedgraph both depend on *engine behavior*, not
/// implementation details. You can swap CPU/Metal/FFI backends under
/// `llama-runtime` without changing the app.
pub trait LlamaEngine: Send + Sync {
    /// Load a model from disk given a specification.
    fn load_model(&self, spec: &ModelSpec) -> Result<ModelHandle>;

    /// Convert text into a sequence of token IDs.
    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;

    /// Convert token IDs back into text.
    fn detokenize(&self, tokens: &[i32]) -> Result<String>;

    /// Run the prefill phase: process prompt tokens and populate the KV cache.
    fn prefill(&self, session: &mut Session, tokens: &[i32]) -> Result<PrefillResult>;

    /// Run the decode phase: stream tokens one at a time from the model.
    fn decode(&self, session: &mut Session) -> Result<TokenStream>;

    /// Generate embeddings for a batch of texts (for oxidizedRAG integration).
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}
