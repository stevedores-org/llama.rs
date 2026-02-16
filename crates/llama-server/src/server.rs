//! Server setup and routing.

use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;

use crate::{handlers, state::AppState};

/// Create the API router with all routes.
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route(
            "/v1/chat/completions",
            post(handlers::chat::handle_chat_completion),
        )
        .route(
            "/v1/embeddings",
            post(handlers::embeddings::handle_embeddings),
        )
        .route("/health", get(handlers::health::handle_health))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Run the HTTP server.
pub async fn run_server(
    state: AppState,
    addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
