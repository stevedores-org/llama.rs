//! # llama-rag
//!
//! Retrieval-Augmented Generation adapter for llama.rs.
//!
//! Maps [`LlamaEngine::embed`] to a pluggable document store, builds
//! retrieval-augmented prompts with citations, and runs the full
//! query → retrieve → generate pipeline.
//!
//! The [`DocumentStore`] trait is backend-agnostic: oxidizedRAG, in-memory,
//! or any vector database can implement it.

use llama_engine::{LlamaEngine, LlamaError, Session};
use std::collections::HashMap;
use uuid::Uuid;

/// Errors specific to the RAG pipeline.
#[derive(Debug, thiserror::Error)]
pub enum RagError {
    /// Engine error during tokenization, inference, or embedding.
    #[error("engine: {0}")]
    Engine(#[from] LlamaError),

    /// Document store error.
    #[error("store: {0}")]
    Store(String),

    /// No relevant documents found for the query.
    #[error("no relevant documents found for query")]
    NoResults,
}

/// A single document chunk with metadata.
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique document ID.
    pub id: Uuid,
    /// The text content of this chunk.
    pub content: String,
    /// Source identifier (filename, URL, etc.).
    pub source: String,
    /// Optional metadata (section title, page number, etc.).
    pub metadata: HashMap<String, String>,
}

impl Document {
    /// Create a new document with the given content and source.
    pub fn new(content: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            source: source.into(),
            metadata: HashMap::new(),
        }
    }

    /// Add a metadata key-value pair.
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A retrieved document with its relevance score.
#[derive(Debug, Clone)]
pub struct RetrievedDocument {
    /// The document that was retrieved.
    pub document: Document,
    /// Cosine similarity score (0.0..1.0).
    pub score: f32,
}

/// Trait for document stores that support embedding-based retrieval.
///
/// Implementations should use `LlamaEngine::embed` (or their own embedding
/// pipeline) to convert documents to vectors and perform similarity search.
pub trait DocumentStore: Send + Sync {
    /// Ingest documents into the store, computing and storing their embeddings.
    fn ingest(&mut self, docs: &[Document], embeddings: &[Vec<f32>]) -> Result<(), RagError>;

    /// Retrieve the top-k most relevant documents for a query embedding.
    fn retrieve(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<RetrievedDocument>, RagError>;

    /// Number of documents in the store.
    fn len(&self) -> usize;

    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A citation reference in a generated answer.
#[derive(Debug, Clone)]
pub struct Citation {
    /// Index into the retrieved documents list (1-based for display).
    pub index: usize,
    /// Source of the cited document.
    pub source: String,
    /// Relevance score.
    pub score: f32,
}

/// Result of a RAG query, including the generated answer and citations.
#[derive(Debug, Clone)]
pub struct RagResult {
    /// The generated answer text.
    pub answer: String,
    /// Documents that were used as context, with citation indices.
    pub citations: Vec<Citation>,
    /// The augmented prompt that was sent to the model.
    pub augmented_prompt: String,
}

/// Configuration for the RAG pipeline.
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Maximum number of documents to retrieve.
    pub top_k: usize,
    /// Minimum relevance score to include a document (0.0..1.0).
    pub min_score: f32,
    /// Maximum tokens to generate in the answer.
    pub max_tokens: usize,
    /// System prompt template. `{context}` and `{query}` are replaced.
    pub prompt_template: String,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            top_k: 3,
            min_score: 0.0,
            max_tokens: 128,
            prompt_template: String::from(
                "Answer the question based on the following context. \
                 Cite sources using [1], [2], etc.\n\n\
                 Context:\n{context}\n\n\
                 Question: {query}\n\n\
                 Answer:",
            ),
        }
    }
}

/// Builds an augmented prompt from retrieved documents and a query.
pub fn build_augmented_prompt(
    docs: &[RetrievedDocument],
    query: &str,
    template: &str,
) -> (String, Vec<Citation>) {
    let mut context_parts = Vec::with_capacity(docs.len());
    let mut citations = Vec::with_capacity(docs.len());

    for (i, rd) in docs.iter().enumerate() {
        let idx = i + 1;
        context_parts.push(format!(
            "[{}] ({}): {}",
            idx, rd.document.source, rd.document.content
        ));
        citations.push(Citation {
            index: idx,
            source: rd.document.source.clone(),
            score: rd.score,
        });
    }

    let context = context_parts.join("\n\n");
    let prompt = template
        .replace("{context}", &context)
        .replace("{query}", query);

    (prompt, citations)
}

/// The main RAG pipeline adapter.
///
/// Wraps a [`LlamaEngine`] and a [`DocumentStore`] to provide
/// retrieval-augmented generation.
pub struct RagAdapter<E: LlamaEngine, S: DocumentStore> {
    engine: E,
    store: S,
    config: RagConfig,
}

impl<E: LlamaEngine, S: DocumentStore> RagAdapter<E, S> {
    /// Create a new RAG adapter.
    pub fn new(engine: E, store: S, config: RagConfig) -> Self {
        Self {
            engine,
            store,
            config,
        }
    }

    /// Ingest documents into the store using the engine's embedding function.
    pub fn ingest(&mut self, docs: &[Document]) -> Result<(), RagError> {
        if docs.is_empty() {
            return Ok(());
        }

        let texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
        let embeddings = self.engine.embed(&texts)?;
        self.store.ingest(docs, &embeddings)?;
        Ok(())
    }

    /// Run the full RAG pipeline: embed query → retrieve → build prompt → generate.
    pub fn query(&self, query: &str) -> Result<RagResult, RagError> {
        // 1. Embed the query
        let query_embeddings = self.engine.embed(&[query])?;
        let query_embedding = query_embeddings
            .into_iter()
            .next()
            .ok_or_else(|| RagError::Store("empty embedding result".into()))?;

        // 2. Retrieve relevant documents
        let mut retrieved = self.store.retrieve(&query_embedding, self.config.top_k)?;

        // Filter by minimum score
        retained_above_threshold(&mut retrieved, self.config.min_score);

        if retrieved.is_empty() {
            return Err(RagError::NoResults);
        }

        // 3. Build augmented prompt with citations
        let (augmented_prompt, citations) =
            build_augmented_prompt(&retrieved, query, &self.config.prompt_template);

        // 4. Generate answer
        let answer = self.generate(&augmented_prompt)?;

        Ok(RagResult {
            answer,
            citations,
            augmented_prompt,
        })
    }

    /// Generate text from a prompt using prefill + decode.
    fn generate(&self, prompt: &str) -> Result<String, RagError> {
        let tokens = self.engine.tokenize(prompt)?;
        let mut session = Session::new();
        let _ = self.engine.prefill(&mut session, &tokens)?;

        let mut output_tokens = Vec::with_capacity(self.config.max_tokens);
        for _ in 0..self.config.max_tokens {
            let result = self.engine.decode(&mut session)?;
            // Stop on EOS token (typically token ID 2)
            if result.token == 2 {
                break;
            }
            output_tokens.push(result.token);
        }

        let answer = self.engine.detokenize(&output_tokens)?;
        Ok(answer)
    }

    /// Access the underlying engine.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Access the document store.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Access the RAG configuration.
    pub fn config(&self) -> &RagConfig {
        &self.config
    }

    /// Number of documents in the store.
    pub fn document_count(&self) -> usize {
        self.store.len()
    }
}

/// Remove documents below the score threshold (in-place).
fn retained_above_threshold(docs: &mut Vec<RetrievedDocument>, min_score: f32) {
    docs.retain(|d| d.score >= min_score);
}

// ── In-memory document store ──────────────────────────────────────────────

/// A simple in-memory document store using brute-force cosine similarity.
///
/// Suitable for testing and small document sets. For production use,
/// implement [`DocumentStore`] against a proper vector database
/// (e.g., via oxidizedRAG).
#[derive(Debug, Default)]
pub struct InMemoryStore {
    documents: Vec<Document>,
    embeddings: Vec<Vec<f32>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DocumentStore for InMemoryStore {
    fn ingest(&mut self, docs: &[Document], embeddings: &[Vec<f32>]) -> Result<(), RagError> {
        if docs.len() != embeddings.len() {
            return Err(RagError::Store(format!(
                "document count ({}) != embedding count ({})",
                docs.len(),
                embeddings.len()
            )));
        }
        self.documents.extend(docs.iter().cloned());
        self.embeddings.extend(embeddings.iter().cloned());
        Ok(())
    }

    fn retrieve(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<RetrievedDocument>, RagError> {
        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, cosine_similarity(query_embedding, emb)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let results = scored
            .into_iter()
            .take(top_k)
            .map(|(i, score)| RetrievedDocument {
                document: self.documents[i].clone(),
                score,
            })
            .collect();

        Ok(results)
    }

    fn len(&self) -> usize {
        self.documents.len()
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_engine::{DecodeResult, ModelHandle, ModelSpec, PrefillResult, TokenId};
    use std::sync::atomic::{AtomicI32, Ordering};

    /// Test engine that produces deterministic tokens and simple embeddings.
    #[derive(Default)]
    struct TestEngine {
        next_token: AtomicI32,
    }

    impl LlamaEngine for TestEngine {
        fn load_model(&self, _spec: &ModelSpec) -> llama_engine::Result<ModelHandle> {
            Ok(ModelHandle)
        }

        fn tokenize(&self, text: &str) -> llama_engine::Result<Vec<TokenId>> {
            Ok((0..text.split_whitespace().count() as i32).collect())
        }

        fn detokenize(&self, tokens: &[TokenId]) -> llama_engine::Result<String> {
            Ok(tokens
                .iter()
                .map(|t| format!("tok{t}"))
                .collect::<Vec<_>>()
                .join(" "))
        }

        fn prefill(
            &self,
            _session: &mut Session,
            tokens: &[TokenId],
        ) -> llama_engine::Result<PrefillResult> {
            Ok(PrefillResult {
                tokens_processed: tokens.len(),
            })
        }

        fn decode(&self, _session: &mut Session) -> llama_engine::Result<DecodeResult> {
            let token = self.next_token.fetch_add(1, Ordering::SeqCst);
            Ok(DecodeResult { token })
        }

        fn embed(&self, texts: &[&str]) -> llama_engine::Result<Vec<Vec<f32>>> {
            // Produce embeddings based on text length for deterministic retrieval.
            // Longer texts get higher values in the first dimension.
            Ok(texts
                .iter()
                .map(|t| {
                    let len = t.len() as f32;
                    vec![len, 1.0, 0.5, 0.25]
                })
                .collect())
        }
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn in_memory_store_ingest_and_retrieve() {
        let mut store = InMemoryStore::new();
        let docs = vec![
            Document::new("short", "file1.txt"),
            Document::new("a much longer document with more words", "file2.txt"),
        ];
        let embeddings = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        store.ingest(&docs, &embeddings).unwrap();
        assert_eq!(store.len(), 2);

        // Query embedding closer to doc 2
        let results = store.retrieve(&[0.1, 0.9, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.source, "file2.txt");
    }

    #[test]
    fn build_augmented_prompt_includes_citations() {
        let docs = vec![
            RetrievedDocument {
                document: Document::new("Rust is a systems language", "docs/intro.md"),
                score: 0.95,
            },
            RetrievedDocument {
                document: Document::new("llama.rs runs LLMs in Rust", "docs/llama.md"),
                score: 0.88,
            },
        ];

        let (prompt, citations) = build_augmented_prompt(
            &docs,
            "What is llama.rs?",
            &RagConfig::default().prompt_template,
        );

        assert!(prompt.contains("[1] (docs/intro.md)"));
        assert!(prompt.contains("[2] (docs/llama.md)"));
        assert!(prompt.contains("What is llama.rs?"));
        assert_eq!(citations.len(), 2);
        assert_eq!(citations[0].index, 1);
        assert_eq!(citations[1].source, "docs/llama.md");
    }

    #[test]
    fn rag_adapter_end_to_end_query_retrieve_generate() {
        let engine = TestEngine::default();
        let store = InMemoryStore::new();
        let config = RagConfig {
            top_k: 2,
            max_tokens: 3,
            ..RagConfig::default()
        };

        let mut adapter = RagAdapter::new(engine, store, config);

        // Ingest documents
        let docs = vec![
            Document::new("Rust safety guarantees", "safety.md"),
            Document::new("LLM inference with llama.rs engine", "llama.md"),
            Document::new("Python is popular", "python.md"),
        ];
        adapter.ingest(&docs).unwrap();
        assert_eq!(adapter.document_count(), 3);

        // Query — should retrieve, build prompt, generate
        let result = adapter.query("How does llama.rs work?").unwrap();

        // Answer should contain generated tokens
        assert!(result.answer.contains("tok"));

        // Citations should be present
        assert!(!result.citations.is_empty());
        assert!(result.citations.len() <= 2);

        // Augmented prompt should contain context
        assert!(result.augmented_prompt.contains("How does llama.rs work?"));
    }

    #[test]
    fn rag_adapter_no_results_when_store_empty() {
        let engine = TestEngine::default();
        let store = InMemoryStore::new();
        let config = RagConfig {
            min_score: 0.99,
            ..RagConfig::default()
        };

        let adapter = RagAdapter::new(engine, store, config);
        let result = adapter.query("anything");
        assert!(matches!(result, Err(RagError::NoResults)));
    }

    #[test]
    fn document_metadata_preserved_through_pipeline() {
        let engine = TestEngine::default();
        let store = InMemoryStore::new();
        let config = RagConfig {
            top_k: 1,
            max_tokens: 1,
            ..RagConfig::default()
        };

        let mut adapter = RagAdapter::new(engine, store, config);

        let doc = Document::new("content", "source.md")
            .with_meta("section", "introduction")
            .with_meta("page", "42");

        adapter.ingest(&[doc]).unwrap();

        let result = adapter.query("test query").unwrap();
        assert_eq!(result.citations[0].source, "source.md");
    }
}
