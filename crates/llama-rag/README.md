# llama-rag: RAG Adapter for llama.rs

Bridges **llama.rs** inference capabilities with **oxidizedRAG**'s retrieval-augmented generation system.

## Features

- **Async trait implementations** for oxidizedRAG integration:
  - `AsyncEmbedder`: Text-to-vector embeddings via `LlamaEngine`
  - `AsyncLanguageModel`: Text generation for RAG prompts
- **Synchronous adapters** for compatibility:
  - `LlamaEmbedder`: Implements `Embedder` trait
  - `RagPipeline`: End-to-end query → retrieve → generate
  - `RagPromptBuilder`: Builds prompts with inline citations
- **Zero external API dependencies**: All inference runs locally with Metal/CPU backends
- **Citation tracking**: Automatic source attribution in generated responses

## Quick Start

### Basic Setup

```rust
use llama_rag::{LlamaAsyncEmbedder, LlamaAsyncLanguageModel, RagConfig};
use llama_runtime::MockEngine;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let engine = Arc::new(MockEngine::new());

    // Create async embedder
    let config = RagConfig::new().with_embedding_dim(128);
    let embedder = LlamaAsyncEmbedder::new(engine.clone(), config).await?;

    // Create async language model
    let llm = LlamaAsyncLanguageModel::new(
        engine,
        RagConfig::new().with_max_tokens(512)
    );

    // Use with oxidizedRAG
    let embedding = embedder.embed("Hello world").await?;
    println!("Embedding dimension: {}", embedding.len());

    let response = llm.complete("What is RAG?").await?;
    println!("Response: {}", response);

    Ok(())
}
```

### With oxidizedRAG

```rust
use llama_rag::{LlamaAsyncEmbedder, LlamaAsyncLanguageModel, RagConfig};
use graphrag_core::core::traits::{AsyncEmbedder, AsyncLanguageModel};
use std::sync::Arc;

// Create adapters
let embedder = Box::new(
    LlamaAsyncEmbedder::new(engine, RagConfig::default()).await?
) as Box<dyn AsyncEmbedder>;

let llm = Box::new(
    LlamaAsyncLanguageModel::new(engine, RagConfig::default())
) as Box<dyn AsyncLanguageModel>;

// Use with GraphRAG pipeline
let mut graphrag = oxidizedRAG::GraphRAG::builder()
    .with_embedder(embedder)
    .with_llm(llm)
    .build()?;
```

## Configuration

`RagConfig` controls adapter behavior:

```rust
let config = RagConfig::default()
    .with_embedding_dim(384)           // Embedding dimension
    .with_max_tokens(1024)              // Max generation tokens
    .with_temperature(0.7)              // Sampling temperature
    .with_max_concurrent_embeds(10)     // Concurrent embedding requests
    .with_embed_batch_size(32);         // Batch size for embeddings
```

## API Reference

### `LlamaAsyncEmbedder`

Implements `AsyncEmbedder` trait:

- `async fn embed(&self, text: &str) -> Result<Vec<f32>>` - Single embedding
- `async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>` - Batch embeddings
- `async fn embed_batch_concurrent(&self, texts: &[&str], max_concurrent: usize) -> Result<Vec<Vec<f32>>>` - Concurrent batch
- `fn dimension(&self) -> usize` - Embedding dimensionality
- `async fn is_ready(&self) -> bool` - Check readiness
- `async fn health_check(&self) -> Result<bool>` - Health status

### `LlamaAsyncLanguageModel`

Implements `AsyncLanguageModel` trait:

- `async fn complete(&self, prompt: &str) -> Result<String>` - Text completion
- `async fn complete_with_params(&self, prompt: &str, params: GenerationParams) -> Result<String>` - With parameters
- `async fn complete_batch(&self, prompts: &[&str]) -> Result<Vec<String>>` - Batch completions
- `async fn complete_batch_concurrent(&self, prompts: &[&str], max_concurrent: usize) -> Result<Vec<String>>` - Concurrent batch
- `async fn complete_streaming(&self, prompt: &str) -> Result<Stream<...>>` - Streaming (MVP: buffered)
- `async fn is_available(&self) -> bool` - Model availability
- `async fn model_info(&self) -> ModelInfo` - Model metadata
- `async fn estimate_tokens(&self, prompt: &str) -> Result<usize>` - Token estimation

## Synchronous API (Legacy)

For compatibility, synchronous adapters are still available:

```rust
use llama_rag::LlamaEmbedder;

let embedder = LlamaEmbedder::new(engine, 384);
let embedding = embedder.embed("Hello")?;
```

## RAG Pipeline

The synchronous `RagPipeline` provides end-to-end query processing:

```rust
use llama_rag::RagPipeline;

let pipeline = RagPipeline::new(engine, 384)
    .max_generate_tokens(256)
    .max_context_chars(2048);

// Get query embedding for vector search
let query_vec = pipeline.embed_query("What is X?")?;

// Search vector store for relevant contexts...
// Then generate with context
let result = pipeline.generate_with_context("What is X?", contexts)?;

println!("Generated: {}", result.text);
println!("Citations: {:?}", result.citations);
println!("Source documents: {:?}", result.citations.iter()
    .map(|c| &c.source_id).collect::<Vec<_>>());
```

## Error Handling

All adapters return `GraphRAGError` for compatibility with oxidizedRAG:

```rust
use graphrag_core::core::GraphRAGError;

match embedder.embed("test").await {
    Ok(vec) => println!("Embedding: {:?}", vec),
    Err(GraphRAGError::Embedding { message }) => {
        eprintln!("Embedding failed: {}", message);
    }
    Err(e) => eprintln!("Error: {:?}", e),
}
```

## Performance Characteristics

- **Embedding**: Non-blocking async via `tokio::task::spawn_blocking`
- **Generation**: Non-blocking async via `tokio::task::spawn_blocking`
- **Concurrency**: Fully async with `futures::stream::buffer_unordered`
- **Memory**: Arc-wrapped engine enables zero-copy shared access
- **CPU bound**: Uses blocking task pool to avoid starving async runtime

## Testing

Run all tests:

```bash
cargo test -p llama-rag
```

Tests use `MockEngine` for deterministic behavior. Real inference uses Metal/CPU backends from `llama-runtime`.

## Future Enhancements

- **Streaming**: True token-by-token streaming with async channels
- **Parameter mapping**: Wire temperature/top_p to llama sampler
- **Session pooling**: Reuse sessions for performance
- **Metrics**: Token/latency tracking via `get_usage_stats()`
- **Vector store**: Native HNSW implementation
- **Graph store**: Knowledge graph persistence

## Integration with oxidizedRAG

The adapters are designed to drop in as replacements for Ollama/OpenAI in oxidizedRAG:

```rust
// Before: Using Ollama over HTTP
let embedder = Box::new(OllamaEmbedder::new("http://localhost:11434".into())?);

// After: Using llama.rs locally
let embedder = Box::new(
    LlamaAsyncEmbedder::new(engine, RagConfig::default()).await?
);

// Rest of code is unchanged!
```

## License

Licensed under the same terms as llama.rs.
