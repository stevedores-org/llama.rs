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
pub async fn handle_chat_completion(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    let prompt = format_messages(&req.messages);
    let prompt_tokens = state.engine.tokenize(&prompt)?;
    let max_tokens = req.max_tokens.unwrap_or(state.config.max_tokens);

    if req.stream {
        return Ok(
            streaming::stream_chat_completion(state, prompt_tokens, max_tokens).into_response(),
        );
    }

    // Non-streaming response
    let prompt_token_count = prompt_tokens.len();
    let mut session = Session::new();
    let _ = state.engine.prefill(&mut session, &prompt_tokens)?;

    let mut generated_tokens = Vec::new();
    for _ in 0..max_tokens {
        let result = state.engine.decode(&mut session)?;
        generated_tokens.push(result.token);
        if result.token == 2 {
            break;
        }
    }

    let generated_text = state.engine.detokenize(&generated_tokens)?;
    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = Utc::now().timestamp() as u64;

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
