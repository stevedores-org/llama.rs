//! Async embedder implementation for RAG.

use crate::config::RagConfig;
use crate::error::RagError;
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use graphrag_core::core::traits::AsyncEmbedder;
use graphrag_core::core::GraphRAGError;
use llama_engine::LlamaEngine;
use std::sync::Arc;
use tokio::task;

/// Async embedder wrapper around `LlamaEngine`.
///
/// Implements `AsyncEmbedder` trait for use with oxidizedRAG.
/// Provides both single and batch embedding operations with optional concurrency control.
pub struct LlamaAsyncEmbedder {
    engine: Arc<dyn LlamaEngine>,
    config: RagConfig,
    dimension: usize,
}

impl LlamaAsyncEmbedder {
    /// Create a new async embedder.
    ///
    /// Auto-detects embedding dimension by running a test embedding if not specified in config.
    pub async fn new(engine: Arc<dyn LlamaEngine>, config: RagConfig) -> Result<Self, RagError> {
        // Auto-detect dimension if not specified
        let dimension = match config.embedding_dim {
            Some(dim) => dim,
            None => {
                // Test embed to detect dimension
                let engine_clone = engine.clone();
                let test_vec = task::spawn_blocking(move || engine_clone.embed(&["test"]))
                    .await
                    .map_err(|e| RagError::Embedding(format!("task join: {}", e)))?
                    .map_err(|e| RagError::Embedding(e.to_string()))?;

                test_vec.first().map(|v| v.len()).ok_or_else(|| {
                    RagError::Embedding("engine returned empty embedding".to_string())
                })?
            }
        };

        tracing::debug!("initialized async embedder with dimension {}", dimension);

        Ok(Self {
            engine,
            config,
            dimension,
        })
    }

    /// Create a new embedder with default config.
    pub async fn with_engine(engine: Arc<dyn LlamaEngine>) -> Result<Self, RagError> {
        Self::new(engine, RagConfig::default()).await
    }
}

#[async_trait]
impl AsyncEmbedder for LlamaAsyncEmbedder {
    type Error = GraphRAGError;

    async fn embed(&self, text: &str) -> Result<Vec<f32>, GraphRAGError> {
        let engine = self.engine.clone();
        let text = text.to_string();

        task::spawn_blocking(move || {
            engine
                .embed(&[&text])
                .map(|mut vecs| vecs.remove(0))
                .map_err(|e| RagError::Embedding(e.to_string()))
        })
        .await
        .map_err(|e| GraphRAGError::Retrieval {
            message: format!("task join: {}", e),
        })?
        .map_err(|e: RagError| e.into())
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, GraphRAGError> {
        let engine = self.engine.clone();
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        task::spawn_blocking(move || {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            engine
                .embed(&refs)
                .map_err(|e| RagError::Embedding(e.to_string()))
        })
        .await
        .map_err(|e| GraphRAGError::Retrieval {
            message: format!("task join: {}", e),
        })?
        .map_err(|e: RagError| e.into())
    }

    async fn embed_batch_concurrent(
        &self,
        texts: &[&str],
        max_concurrent: usize,
    ) -> Result<Vec<Vec<f32>>, GraphRAGError> {
        // Chunk texts into batches (convert to owned strings for lifetime safety)
        let batch_size = self.config.embed_batch_size;
        let chunks: Vec<Vec<String>> = texts
            .chunks(batch_size)
            .map(|chunk| chunk.iter().map(|s| s.to_string()).collect())
            .collect();

        // Process chunks concurrently with limit
        let results: Vec<Result<Vec<Vec<f32>>, GraphRAGError>> = stream::iter(chunks)
            .map(|chunk| {
                let engine = self.engine.clone();
                async move {
                    let result = tokio::task::spawn_blocking(move || {
                        let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
                        engine
                            .embed(&refs)
                            .map_err(|e| RagError::Embedding(e.to_string()))
                    })
                    .await
                    .map_err(|e| GraphRAGError::Retrieval {
                        message: format!("task join: {}", e),
                    })?
                    .map_err(|e: RagError| -> GraphRAGError { e.into() })?;
                    Ok(result)
                }
            })
            .buffer_unordered(max_concurrent)
            .collect::<Vec<_>>()
            .await;

        // Flatten results
        let mut embeddings = Vec::new();
        for result in results {
            embeddings.extend(result?);
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn is_ready(&self) -> bool {
        // Engine is always ready if it exists
        true
    }

    async fn health_check(&self) -> Result<bool, GraphRAGError> {
        // Try a test embedding
        self.embed("health check").await.map(|_| true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_runtime::MockEngine;

    #[tokio::test]
    async fn async_embedder_single_text() {
        let engine = Arc::new(MockEngine::new());
        let config = RagConfig::new().with_embedding_dim(128);
        let embedder = LlamaAsyncEmbedder::new(engine, config).await.unwrap();

        let vec = embedder.embed("hello world").await.unwrap();
        assert_eq!(vec.len(), 128);
    }

    #[tokio::test]
    async fn async_embedder_batch() {
        let engine = Arc::new(MockEngine::new());
        let config = RagConfig::new().with_embedding_dim(128);
        let embedder = LlamaAsyncEmbedder::new(engine, config).await.unwrap();

        let vecs = embedder.embed_batch(&["hello", "world"]).await.unwrap();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].len(), 128);
    }

    #[tokio::test]
    async fn async_embedder_concurrent_batch() {
        let engine = Arc::new(MockEngine::new());
        let config = RagConfig::new()
            .with_embedding_dim(128)
            .with_embed_batch_size(1);
        let embedder = LlamaAsyncEmbedder::new(engine, config).await.unwrap();

        let texts = vec!["a", "b", "c", "d"];
        let vecs = embedder.embed_batch_concurrent(&texts, 2).await.unwrap();
        assert_eq!(vecs.len(), 4);
    }

    #[tokio::test]
    async fn async_embedder_dimension() {
        let engine = Arc::new(MockEngine::new());
        let config = RagConfig::new().with_embedding_dim(128);
        let embedder = LlamaAsyncEmbedder::new(engine, config).await.unwrap();
        assert_eq!(embedder.dimension(), 128);
    }

    #[tokio::test]
    async fn async_embedder_is_ready() {
        let engine = Arc::new(MockEngine::new());
        let config = RagConfig::new().with_embedding_dim(128);
        let embedder = LlamaAsyncEmbedder::new(engine, config).await.unwrap();
        assert!(embedder.is_ready().await);
    }

    #[tokio::test]
    async fn async_embedder_health_check() {
        let engine = Arc::new(MockEngine::new());
        let config = RagConfig::new().with_embedding_dim(128);
        let embedder = LlamaAsyncEmbedder::new(engine, config).await.unwrap();
        assert!(embedder.health_check().await.unwrap());
    }
}
