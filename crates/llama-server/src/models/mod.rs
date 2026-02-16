//! OpenAI-compatible request/response types.

pub mod chat;
pub mod common;
pub mod embeddings;
pub mod streaming;

pub use chat::{ChatChoice, ChatCompletionRequest, ChatCompletionResponse};
pub use common::{ChatMessage, Usage};
pub use embeddings::{Embedding, EmbeddingRequest, EmbeddingResponse};
pub use streaming::ChatCompletionChunk;
