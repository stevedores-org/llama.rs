//! End-to-end RAG pipeline: query → retrieve → generate with citations.

use crate::embedder::LlamaEmbedder;
use crate::prompt::{Citation, RagPromptBuilder, RetrievedContext};
use graphrag_core::core::traits::Embedder;
use llama_engine::{LlamaEngine, Session, TokenId};
use std::sync::Arc;

/// Result of a RAG-augmented generation.
#[derive(Debug, Clone)]
pub struct RagResult {
    /// Generated text from the model.
    pub text: String,
    /// Citations included in the prompt context.
    pub citations: Vec<Citation>,
    /// The augmented prompt that was sent to the model.
    pub augmented_prompt: String,
    /// Number of context chunks used.
    pub context_count: usize,
}

/// End-to-end RAG pipeline.
///
/// Workflow:
/// 1. Embed the query using `LlamaEmbedder`
/// 2. Search a document store for relevant chunks (provided externally)
/// 3. Build an augmented prompt with citations
/// 4. Generate a response using the engine
pub struct RagPipeline {
    engine: Arc<dyn LlamaEngine>,
    embedder: LlamaEmbedder,
    max_generate_tokens: usize,
    max_context_chars: usize,
}

impl RagPipeline {
    /// Create a new RAG pipeline.
    pub fn new(engine: Arc<dyn LlamaEngine>, embedding_dim: usize) -> Self {
        let embedder = LlamaEmbedder::new(engine.clone(), embedding_dim);
        Self {
            engine,
            embedder,
            max_generate_tokens: 256,
            max_context_chars: 2048,
        }
    }

    /// Set maximum tokens to generate.
    pub fn max_generate_tokens(mut self, n: usize) -> Self {
        self.max_generate_tokens = n;
        self
    }

    /// Set maximum context characters in the prompt.
    pub fn max_context_chars(mut self, n: usize) -> Self {
        self.max_context_chars = n;
        self
    }

    /// Get a reference to the embedder (for external vector search).
    pub fn embedder(&self) -> &LlamaEmbedder {
        &self.embedder
    }

    /// Execute the RAG pipeline given a query and pre-retrieved contexts.
    ///
    /// The caller is responsible for vector search (using `embedder()` to get
    /// query embeddings, then searching their vector store). This keeps the
    /// pipeline agnostic to the specific vector store implementation.
    pub fn generate_with_context(
        &self,
        query: &str,
        contexts: Vec<RetrievedContext>,
    ) -> Result<RagResult, RagError> {
        let context_count = contexts.len();

        // Build augmented prompt
        let mut builder = RagPromptBuilder::new().max_context_chars(self.max_context_chars);
        for ctx in contexts {
            builder.add_context(ctx);
        }
        let (augmented_prompt, citations) = builder.build(query);

        // Tokenize and generate
        let tokens = self
            .engine
            .tokenize(&augmented_prompt)
            .map_err(|e| RagError::Engine(e.to_string()))?;

        let mut session = Session::new();
        let _ = self
            .engine
            .prefill(&mut session, &tokens)
            .map_err(|e| RagError::Engine(e.to_string()))?;

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        for _ in 0..self.max_generate_tokens {
            let result = self
                .engine
                .decode(&mut session)
                .map_err(|e| RagError::Engine(e.to_string()))?;

            // EOS
            if result.token == 2 {
                break;
            }
            generated_tokens.push(result.token);
        }

        let text = self
            .engine
            .detokenize(&generated_tokens)
            .map_err(|e| RagError::Engine(e.to_string()))?;

        Ok(RagResult {
            text,
            citations,
            augmented_prompt,
            context_count,
        })
    }

    /// Convenience: embed a query for vector search.
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>, RagError> {
        self.embedder
            .embed(query)
            .map_err(|e| RagError::Embedding(e.to_string()))
    }
}

/// Errors from the RAG pipeline.
#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error("engine error: {0}")]
    Engine(String),
    #[error("embedding error: {0}")]
    Embedding(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_runtime::MockEngine;

    #[test]
    fn end_to_end_query_retrieve_generate() {
        let engine = Arc::new(MockEngine::new());
        let pipeline = RagPipeline::new(engine, 128).max_generate_tokens(3);

        // Simulate retrieved contexts (would come from vector search in production)
        let contexts = vec![
            RetrievedContext {
                content: "The quick brown fox jumps over the lazy dog.".to_string(),
                source_id: "doc-animals-1".to_string(),
                score: 0.92,
            },
            RetrievedContext {
                content: "Foxes are omnivorous mammals belonging to the family Canidae."
                    .to_string(),
                source_id: "doc-animals-2".to_string(),
                score: 0.85,
            },
        ];

        let result = pipeline
            .generate_with_context("What do foxes eat?", contexts)
            .unwrap();

        // Verify all parts of the result
        assert!(!result.text.is_empty(), "should generate some text");
        assert_eq!(result.citations.len(), 2, "should have 2 citations");
        assert_eq!(result.context_count, 2);
        assert!(
            result.augmented_prompt.contains("[1]"),
            "prompt should have citation markers"
        );
        assert!(
            result.augmented_prompt.contains("What do foxes eat?"),
            "prompt should contain original query"
        );
        assert!(
            result.augmented_prompt.contains("Cite sources"),
            "prompt should instruct model to cite"
        );

        // Citations should be sorted by score (highest first)
        assert!(result.citations[0].score >= result.citations[1].score);
        assert_eq!(result.citations[0].source_id, "doc-animals-1");
        assert_eq!(result.citations[1].source_id, "doc-animals-2");
    }

    #[test]
    fn embed_query_returns_vector() {
        let engine = Arc::new(MockEngine::new());
        let pipeline = RagPipeline::new(engine, 128);

        let embedding = pipeline.embed_query("test query").unwrap();
        assert_eq!(embedding.len(), 128);
    }

    #[test]
    fn empty_context_still_generates() {
        let engine = Arc::new(MockEngine::new());
        let pipeline = RagPipeline::new(engine, 128).max_generate_tokens(2);

        let result = pipeline
            .generate_with_context("hello world", vec![])
            .unwrap();
        assert!(result.citations.is_empty());
        assert_eq!(result.context_count, 0);
    }
}
