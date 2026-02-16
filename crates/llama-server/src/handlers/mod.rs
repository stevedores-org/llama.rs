//! HTTP request handlers for API endpoints.

pub mod chat;
pub mod embeddings;
pub mod health;

pub use chat::handle_chat_completion;
pub use embeddings::handle_embeddings;
pub use health::handle_health;
