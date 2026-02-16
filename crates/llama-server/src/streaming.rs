//! Server-Sent Events (SSE) streaming for chat completions.
//!
//! Implements the OpenAI-compatible streaming protocol:
//! - Each chunk is sent as `data: {json}\n\n`
//! - Final message is `data: [DONE]\n\n`
//! - Stream stops immediately when the client disconnects (via CancellationToken).

use axum::response::sse::{Event, Sse};
use chrono::Utc;
use futures::stream::Stream;
use llama_engine::Session;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::models::streaming::{ChatChoiceDelta, ChatCompletionChunk, ChatDelta};
use crate::session_manager::SessionGuard;
use crate::state::AppState;

/// Create an SSE stream that generates chat completion chunks.
///
/// The stream owns the `SessionGuard`. When the client disconnects, axum drops
/// the stream, which drops the guard, which cancels the token and frees the
/// session slot + semaphore permit.
pub fn stream_chat_completion(
    state: AppState,
    prompt_tokens: Vec<llama_engine::TokenId>,
    max_tokens: usize,
    cancel: CancellationToken,
    guard: SessionGuard,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = Utc::now().timestamp() as u64;
    let model = state.config.model_name.clone();
    let engine = state.engine.clone();

    let stream = async_stream::stream! {
        // Keep guard alive for the lifetime of the stream.
        let _guard = guard;

        // Initial chunk: role announcement
        let role_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatChoiceDelta {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&role_chunk).unwrap()));

        // Decode loop: generate tokens and stream as content deltas
        let mut session = Session::new();
        let _ = engine.prefill(&mut session, &prompt_tokens);

        let mut finish_reason = "stop";
        for i in 0..max_tokens {
            // Check cancellation before each decode step.
            if cancel.is_cancelled() {
                tracing::debug!("stream cancelled by client disconnect");
                return;
            }

            let result = match engine.decode(&mut session) {
                Ok(r) => r,
                Err(_) => {
                    finish_reason = "stop";
                    break;
                }
            };

            // EOS check
            if result.token == 2 {
                break;
            }

            // Detokenize single token
            let text = match engine.detokenize(&[result.token]) {
                Ok(t) => t,
                Err(_) => break,
            };

            let content_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChatChoiceDelta {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: Some(text),
                    },
                    finish_reason: None,
                }],
            };
            yield Ok(Event::default().data(serde_json::to_string(&content_chunk).unwrap()));

            // Check if we've hit the limit
            if i + 1 >= max_tokens {
                finish_reason = "length";
            }
        }

        // Final chunk: finish reason (only if not cancelled)
        if !cancel.is_cancelled() {
            let final_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChatChoiceDelta {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some(finish_reason.to_string()),
                }],
            };
            yield Ok(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));

            // [DONE] sentinel
            yield Ok(Event::default().data("[DONE]"));
        }
    };

    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default())
}
