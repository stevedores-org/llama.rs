//! Server-Sent Events (SSE) streaming types for chat completions.

use serde::{Deserialize, Serialize};

/// Chat completion chunk for streaming responses.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceDelta>,
}

/// Chat choice with delta for streaming.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatChoiceDelta {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

/// Delta object containing incremental content.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}
