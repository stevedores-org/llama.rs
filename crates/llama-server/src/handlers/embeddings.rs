//! Embeddings handler.

use axum::{extract::State, Json};

use crate::{
    error::ServerError,
    models::{embeddings::EmbeddingInput, Embedding, EmbeddingRequest, EmbeddingResponse, Usage},
    state::AppState,
};

/// Handle embedding requests.
pub async fn handle_embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ServerError> {
    // Convert input to list of strings
    let texts: Vec<&str> = match &req.input {
        EmbeddingInput::Single(s) => vec![s.as_str()],
        EmbeddingInput::Batch(v) => v.iter().map(|s| s.as_str()).collect(),
    };

    // Get embeddings from engine
    let embeddings = state.engine.embed(&texts)?;

    // Create embedding response objects
    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| Embedding {
            object: "embedding".to_string(),
            embedding,
            index,
        })
        .collect();

    // Calculate token counts using the engine's tokenizer for accuracy.
    let prompt_tokens: usize = texts
        .iter()
        .map(|t| state.engine.tokenize(t).map(|v| v.len()).unwrap_or(0))
        .sum();

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: req.model,
        usage: Usage {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
        },
    }))
}
