//! Health check handler.

use axum::Json;
use serde_json::{json, Value};

/// Handle health check requests.
pub async fn handle_health() -> Json<Value> {
    Json(json!({
        "status": "ok"
    }))
}
