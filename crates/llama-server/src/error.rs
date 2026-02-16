//! HTTP error handling and response mapping.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use llama_engine::LlamaError;
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("llama error: {0}")]
    LlamaError(#[from] LlamaError),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("server at capacity")]
    ServiceUnavailable,
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match self {
            ServerError::LlamaError(LlamaError::Tokenization(msg)) => {
                (StatusCode::BAD_REQUEST, "invalid_request_error", msg)
            }
            ServerError::InvalidRequest(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_request_error", msg)
            }
            ServerError::LlamaError(LlamaError::ModelLoad(msg)) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "server_error", msg)
            }
            ServerError::LlamaError(LlamaError::Inference(msg)) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "server_error", msg)
            }
            ServerError::ServiceUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "Server at capacity, try again later".to_string(),
            ),
        };

        let body = Json(json!({
            "error": {
                "message": message,
                "type": error_type,
                "param": null,
                "code": null,
            }
        }));

        (status, body).into_response()
    }
}
