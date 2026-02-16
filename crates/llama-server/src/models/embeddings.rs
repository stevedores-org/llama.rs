//! Embedding request/response types.

use crate::models::common::Usage;
use serde::{Deserialize, Serialize};

/// Embedding input (single or batch).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

/// Embedding request.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
}

/// Embedding object.
#[derive(Debug, Serialize)]
pub struct Embedding {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// Embedding response.
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<Embedding>,
    pub model: String,
    pub usage: Usage,
}
