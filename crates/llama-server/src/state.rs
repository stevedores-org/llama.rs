//! Application state and configuration.

use crate::session_manager::SessionManager;
use llama_engine::LlamaEngine;
use std::sync::Arc;

/// Application state shared across handlers.
#[derive(Clone)]
pub struct AppState {
    /// Shared engine for inference.
    pub engine: Arc<dyn LlamaEngine>,
    /// Server configuration.
    pub config: ServerConfig,
    /// Session lifecycle manager (concurrency + cancellation).
    pub sessions: Arc<SessionManager>,
}

/// Server configuration parameters.
#[derive(Clone)]
pub struct ServerConfig {
    /// Model name to report in API responses.
    pub model_name: String,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Default temperature for sampling.
    pub default_temperature: f32,
    /// Maximum concurrent inference sessions.
    pub max_concurrent_sessions: usize,
    /// End-of-sequence token ID (model-specific, e.g. 2 for Llama).
    pub eos_token_id: llama_engine::TokenId,
}
