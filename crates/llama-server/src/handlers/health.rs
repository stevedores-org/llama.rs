//! Health check handler.

use axum::{extract::State, Json};
use serde_json::{json, Value};

use crate::state::AppState;

/// Handle health check requests. Includes session utilization stats.
pub async fn handle_health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "sessions": {
            "active": state.sessions.active_count().await,
            "max_concurrent": state.sessions.max_concurrent(),
            "available": state.sessions.available_permits(),
        }
    }))
}
