//! Configuration for llama-rag adapters.

/// Configuration for RAG operations.
#[derive(Clone, Debug)]
pub struct RagConfig {
    /// Embedding dimension (auto-detected if None).
    pub embedding_dim: Option<usize>,

    /// Max concurrent embedding requests.
    pub max_concurrent_embeds: usize,

    /// Max tokens for generation.
    pub max_tokens: usize,

    /// Default temperature for sampling (0.0 = deterministic, 1.0 = random).
    pub temperature: f32,

    /// Batch size for embeddings.
    pub embed_batch_size: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            embedding_dim: None,
            max_concurrent_embeds: 10,
            max_tokens: 512,
            temperature: 0.7,
            embed_batch_size: 32,
        }
    }
}

impl RagConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set embedding dimension.
    pub fn with_embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = Some(dim);
        self
    }

    /// Set max concurrent embeddings.
    pub fn with_max_concurrent_embeds(mut self, n: usize) -> Self {
        self.max_concurrent_embeds = n;
        self
    }

    /// Set max generation tokens.
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Set embedding batch size.
    pub fn with_embed_batch_size(mut self, n: usize) -> Self {
        self.embed_batch_size = n;
        self
    }
}
