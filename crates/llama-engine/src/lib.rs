//! # llama-engine
//!
//! The "narrow waist" of the llama.rs stack. Defines the core [`LlamaEngine`] trait
//! and associated types that all other crates depend on. Implementations can swap
//! CPU/Metal/FFI backends without changing application code.
//!
//! ## Design Notes
//!
//! ### Interior Mutability
//! `LlamaEngine` methods take `&self` (not `&mut self`) to allow shared access across
//! multiple sessions and to enable concurrent inference without synchronizing access.
//! Backends using interior mutability (e.g., `Mutex`, `Arc<RwLock>`) are responsible
//! for thread-safe state management.
//!
//! ### Token Type
//! `TokenId` is aliased as `i32` for FFI compatibility, though token IDs are logically
//! non-negative. This will be reconsidered if a u32/usize conversion barrier emerges.

pub type Result<T> = std::result::Result<T, LlamaError>;

/// Token ID type (i32 for FFI compat; logically non-negative).
pub type TokenId = i32;

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
pub struct ModelHandle;

/// Represents an active inference session with its own KV cache state.
///
/// Sessions hold runtime state (KV cache, token history, etc.) that persists
/// across prefill and decode phases. Multiple sessions can exist simultaneously,
/// each with its own independent state.
///
/// Sessions are intentionally not `Clone` — cloning would imply duplicating
/// KV cache state, which is not a cheap or well-defined operation.
#[derive(Debug)]
pub struct Session {
    /// Unique session ID for tracking and logging.
    pub id: uuid::Uuid,
}

impl Session {
    /// Create a new inference session with a random UUID.
    pub fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
        }
    }

    /// Create a session with an explicit ID (useful for testing/replay).
    pub fn with_id(id: uuid::Uuid) -> Self {
        Self { id }
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of the prefill phase (prompt processing).
#[derive(Debug, Clone)]
pub struct PrefillResult {
    /// Number of tokens processed.
    pub tokens_processed: usize,
}

/// Result of a single decode step.
#[derive(Debug, Clone)]
pub struct DecodeResult {
    /// The decoded token.
    pub token: TokenId,
}

/// The core engine trait — everything else plugs into this.
///
/// Implementations provide inference, tokenization, and embedding functionality.
/// oxidizedRAG and oxidizedgraph depend on *engine behavior*, not implementation
/// details. Swap CPU/Metal/FFI backends without changing application code.
pub trait LlamaEngine: Send + Sync {
    /// Load a model from disk given a specification.
    fn load_model(&self, spec: &ModelSpec) -> Result<ModelHandle>;

    /// Convert text into a sequence of token IDs.
    fn tokenize(&self, text: &str) -> Result<Vec<TokenId>>;

    /// Convert token IDs back into text.
    fn detokenize(&self, tokens: &[TokenId]) -> Result<String>;

    /// Run the prefill phase: process prompt tokens and populate the KV cache.
    fn prefill(&self, session: &mut Session, tokens: &[TokenId]) -> Result<PrefillResult>;

    /// Run the decode phase: produce the next token from the model.
    fn decode(&self, session: &mut Session) -> Result<DecodeResult>;

    /// Generate embeddings for a batch of texts (for oxidizedRAG integration).
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}
