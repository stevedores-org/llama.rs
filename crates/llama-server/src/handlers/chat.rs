//! Chat completion handler.

use axum::{extract::State, response::IntoResponse, Json};
use chrono::Utc;
use uuid::Uuid;

use crate::{
    error::ServerError,
    models::common::ChatMessage,
    models::{ChatChoice, ChatCompletionRequest, ChatCompletionResponse, Usage},
    state::AppState,
    streaming,
};
use llama_engine::Session;

/// Handle chat completion requests (streaming and non-streaming).
///
/// Each request acquires a session guard from the SessionManager, which:
/// - Enforces concurrency limits (returns 503 if at capacity)
/// - Provides a CancellationToken for stream termination
/// - Automatically frees session resources on drop (including client disconnect)
pub async fn handle_chat_completion(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    let prompt = format_messages(&req.messages);
    let prompt_tokens = state.engine.tokenize(&prompt)?;
    let max_tokens = req.max_tokens.unwrap_or(state.config.max_tokens);

    let session_id = Uuid::new_v4();
    let guard = state
        .sessions
        .try_acquire(session_id)
        .await
        .ok_or(ServerError::ServiceUnavailable)?;

    if req.stream {
        let cancel = guard.cancellation_token();
        // Guard is moved into the stream — it stays alive until the stream is dropped.
        // When the client disconnects, axum drops the stream, dropping the guard,
        // which cancels the token and frees the session slot.
        return Ok(streaming::stream_chat_completion(
            state,
            prompt_tokens,
            max_tokens,
            cancel,
            guard,
        )
        .into_response());
    }

    // Non-streaming response — guard lives for the duration of this function
    let prompt_token_count = prompt_tokens.len();
    let mut session = Session::new();
    let _ = state.engine.prefill(&mut session, &prompt_tokens)?;

    let cancel = guard.cancellation_token();
    let mut generated_tokens = Vec::new();
    for _ in 0..max_tokens {
        if cancel.is_cancelled() {
            break;
        }
        let result = state.engine.decode(&mut session)?;
        generated_tokens.push(result.token);
        if result.token == 2 {
            break;
        }
    }

    let generated_text = state.engine.detokenize(&generated_tokens)?;
    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = Utc::now().timestamp() as u64;

    // Guard dropped here — session freed
    drop(guard);

    Ok(Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: state.config.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: generated_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens: generated_tokens.len(),
            total_tokens: prompt_token_count + generated_tokens.len(),
        },
    })
    .into_response())
}

/// Format chat messages into a single prompt string (MVP implementation).
fn format_messages(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}
