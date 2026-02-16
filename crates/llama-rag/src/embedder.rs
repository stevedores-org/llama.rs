//! Adapter implementing graphrag-core's `Embedder` trait via `LlamaEngine::embed()`.

use graphrag_core::core::traits::Embedder;
use graphrag_core::GraphRAGError;
use llama_engine::LlamaEngine;
use std::sync::Arc;

/// Wraps a `LlamaEngine` to provide embeddings for oxidizedRAG.
///
/// This adapter maps graphrag-core's `Embedder` trait onto `LlamaEngine::embed()`,
/// enabling oxidizedRAG to use llama.rs models for vector search and retrieval.
pub struct LlamaEmbedder {
    engine: Arc<dyn LlamaEngine>,
    dimension: usize,
}

impl LlamaEmbedder {
    /// Create a new embedder wrapping the given engine.
    ///
    /// `dimension` must match the embedding size produced by the engine.
    pub fn new(engine: Arc<dyn LlamaEngine>, dimension: usize) -> Self {
        Self { engine, dimension }
    }
}

impl Embedder for LlamaEmbedder {
    type Error = GraphRAGError;

    fn embed(&self, text: &str) -> graphrag_core::Result<Vec<f32>> {
        let results = self
            .engine
            .embed(&[text])
            .map_err(|e| GraphRAGError::Embedding {
                message: e.to_string(),
            })?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| GraphRAGError::Embedding {
                message: "engine returned no embeddings".to_string(),
            })
    }

    fn embed_batch(&self, texts: &[&str]) -> graphrag_core::Result<Vec<Vec<f32>>> {
        self.engine
            .embed(texts)
            .map_err(|e| GraphRAGError::Embedding {
                message: e.to_string(),
            })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn is_ready(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_runtime::MockEngine;

    #[test]
    fn embedder_single_text() {
        let engine = Arc::new(MockEngine::new());
        let embedder = LlamaEmbedder::new(engine, 128);

        let vec = embedder.embed("hello world").unwrap();
        assert_eq!(vec.len(), 128);
    }

    #[test]
    fn embedder_batch() {
        let engine = Arc::new(MockEngine::new());
        let embedder = LlamaEmbedder::new(engine, 128);

        let vecs = embedder.embed_batch(&["hello", "world"]).unwrap();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].len(), 128);
    }

    #[test]
    fn embedder_dimension() {
        let engine = Arc::new(MockEngine::new());
        let embedder = LlamaEmbedder::new(engine, 128);
        assert_eq!(embedder.dimension(), 128);
    }

    #[test]
    fn embedder_is_ready() {
        let engine = Arc::new(MockEngine::new());
        let embedder = LlamaEmbedder::new(engine, 128);
        assert!(embedder.is_ready());
    }
}
