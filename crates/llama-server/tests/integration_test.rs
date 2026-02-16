use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use llama_runtime::MockEngine;
use llama_server::{create_router, AppState, ServerConfig};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;

fn test_state() -> AppState {
    AppState {
        engine: Arc::new(MockEngine::new()),
        config: ServerConfig {
            model_name: "test-model".to_string(),
            max_tokens: 10,
            default_temperature: 0.7,
        },
    }
}

fn json_request(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap()
}

// -- Health endpoint --

#[tokio::test]
async fn health_returns_ok() {
    let app = create_router(test_state());
    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// -- Chat completions (non-streaming) --

#[tokio::test]
async fn chat_completion_non_streaming() {
    let app = create_router(test_state());
    let req = json_request(
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "the quick brown fox jumps"}],
            "stream": false,
            "max_tokens": 3
        }),
    );
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body);
    assert_eq!(status, StatusCode::OK, "body: {body_str}");
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "test-model");
    assert!(json["choices"].is_array());
    assert!(!json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .is_empty());
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert!(json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(json["id"].as_str().unwrap().starts_with("chatcmpl-"));
}

#[tokio::test]
async fn chat_completion_defaults_stream_false() {
    let app = create_router(test_state());
    let req = json_request(
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "the quick brown fox jumps"}]
        }),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "chat.completion");
}

// -- Chat completions (streaming) --

#[tokio::test]
async fn chat_completion_streaming_returns_sse() {
    let app = create_router(test_state());
    let req = json_request(
        "/v1/chat/completions",
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "the quick brown fox jumps"}],
            "stream": true,
            "max_tokens": 3
        }),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // SSE responses use text/event-stream content type
    let content_type = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        content_type.contains("text/event-stream"),
        "expected text/event-stream, got {content_type}"
    );

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8(body.to_vec()).unwrap();

    // Should contain data: lines with JSON chunks
    assert!(body_str.contains("data: "), "should have SSE data lines");
    assert!(body_str.contains("[DONE]"), "should end with [DONE]");
    assert!(
        body_str.contains("chat.completion.chunk"),
        "should contain chunk objects"
    );

    // Parse individual chunks
    let chunks: Vec<&str> = body_str
        .lines()
        .filter(|l| l.starts_with("data: ") && !l.contains("[DONE]"))
        .collect();

    // At minimum: role chunk + final chunk = 2, plus content chunks
    assert!(
        chunks.len() >= 2,
        "expected at least 2 chunks, got {}",
        chunks.len()
    );

    // First chunk should have role
    let first: Value = serde_json::from_str(chunks[0].strip_prefix("data: ").unwrap()).unwrap();
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");

    // Last data chunk (before [DONE]) should have finish_reason
    let last: Value =
        serde_json::from_str(chunks.last().unwrap().strip_prefix("data: ").unwrap()).unwrap();
    assert!(last["choices"][0]["finish_reason"].is_string());
}

// -- Embeddings --

#[tokio::test]
async fn embeddings_single_input() {
    let app = create_router(test_state());
    let req = json_request(
        "/v1/embeddings",
        json!({
            "model": "test-model",
            "input": "hello world"
        }),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["object"], "list");
    assert_eq!(json["data"].as_array().unwrap().len(), 1);
    assert_eq!(json["data"][0]["object"], "embedding");
    assert!(!json["data"][0]["embedding"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn embeddings_batch_input() {
    let app = create_router(test_state());
    let req = json_request(
        "/v1/embeddings",
        json!({
            "model": "test-model",
            "input": ["hello", "world"]
        }),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["data"].as_array().unwrap().len(), 2);
}

// -- Error handling --

#[tokio::test]
async fn invalid_json_returns_error() {
    let app = create_router(test_state());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("not json"))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert!(resp.status().is_client_error());
}
