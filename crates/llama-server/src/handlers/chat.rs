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
    // Warn about sampling parameters not yet plumbed through to the engine.
    if let Some(t) = req.temperature {
        if (t - 1.0).abs() > f32::EPSILON {
            tracing::warn!(
                temperature = t,
                "temperature parameter accepted but not yet applied to sampling"
            );
        }
    }
    if let Some(p) = req.top_p {
        if (p - 1.0).abs() > f32::EPSILON {
            tracing::warn!(
                top_p = p,
                "top_p parameter accepted but not yet applied to sampling"
            );
        }
    }

    let prompt = format_messages(&req.messages);
    let prompt_tokens = state.engine.tokenize(&prompt)?;
    let max_tokens = req.max_tokens.unwrap_or(state.config.max_tokens);
    let eos_token_id = state.config.eos_token_id;

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
            eos_token_id,
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
    let mut hit_eos = false;
    for _ in 0..max_tokens {
        if cancel.is_cancelled() {
            break;
        }
        let result = state.engine.decode(&mut session)?;
        if result.token == eos_token_id {
            hit_eos = true;
            break;
        }
        generated_tokens.push(result.token);
    }

    let finish_reason = if hit_eos || cancel.is_cancelled() {
        "stop"
    } else {
        "length"
    };

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
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens: generated_tokens.len(),
            total_tokens: prompt_token_count + generated_tokens.len(),
        },
    })
    .into_response())
}

// TODO: Replace with proper chat template support (ChatML, Llama 3, etc.)
// via a `chat_template` field in ServerConfig.
/// Format chat messages into a single prompt string (MVP implementation).
fn format_messages(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}
