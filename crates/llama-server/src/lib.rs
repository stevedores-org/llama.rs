//! # llama-server
//!
//! OpenAI-compatible HTTP API for the llama.rs inference engine.
//!
//! Exposes `LlamaEngine` trait methods through REST endpoints compatible with the OpenAI API.
//! Includes support for streaming completions via Server-Sent Events (SSE).

pub mod error;
pub mod handlers;
pub mod models;
pub mod server;
pub mod session_manager;
pub mod state;
pub mod streaming;

pub use error::ServerError;
pub use server::{create_router, run_server};
pub use session_manager::SessionManager;
pub use state::{AppState, ServerConfig};
