use llama_runtime::MockEngine;
use llama_server::{run_server, AppState, ServerConfig, SessionManager};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Create engine
    let engine = Arc::new(MockEngine::new());

    // Server configuration
    let max_concurrent = 64;
    let config = ServerConfig {
        model_name: "llama-3-8b-mock".to_string(),
        max_tokens: 256,
        default_temperature: 0.7,
        max_concurrent_sessions: max_concurrent,
    };

    // Session manager with concurrency limit
    let sessions = SessionManager::new(max_concurrent);

    // Application state
    let state = AppState {
        engine,
        config,
        sessions,
    };

    // Bind to localhost:8080
    let addr = "127.0.0.1:8080".parse()?;
    tracing::info!("Starting server on {}", addr);

    run_server(state, addr).await?;
    Ok(())
}
