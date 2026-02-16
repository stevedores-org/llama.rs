//! Demonstration of async adapter usage with oxidizedRAG.
//!
//! This example shows how to use `LlamaAsyncEmbedder` and `LlamaAsyncLanguageModel`
//! as drop-in replacements for external API-based adapters in oxidizedRAG.

use graphrag_core::core::traits::{AsyncEmbedder, AsyncLanguageModel};
use llama_rag::{LlamaAsyncEmbedder, LlamaAsyncLanguageModel, RagConfig};
use llama_runtime::MockEngine;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== llama-rag Async Adapter Demo ===\n");

    // Create a mock engine (in production, use CPU/Metal backends from llama-runtime)
    let engine = Arc::new(MockEngine::new());
    println!("✓ Created llama.rs engine\n");

    // Configure adapters
    let config = RagConfig::default()
        .with_embedding_dim(128)
        .with_max_tokens(50)
        .with_max_concurrent_embeds(4);

    println!("Configuration:");
    println!("  - Embedding dim: 128");
    println!("  - Max tokens: 50");
    println!("  - Max concurrent: 4\n");

    // Create async embedder
    println!("Creating async embedder...");
    let embedder = LlamaAsyncEmbedder::new(engine.clone(), config.clone()).await?;
    println!("✓ Async embedder ready");
    println!("  - Dimension: {}\n", embedder.dimension());

    // Create async language model
    println!("Creating async language model...");
    let llm = LlamaAsyncLanguageModel::new(engine.clone(), config);
    println!("✓ Async LLM ready");
    println!("  - Model: {}\n", llm.model_info().await.name);

    // Example 1: Single embedding
    println!("--- Example 1: Single Embedding ---");
    let text = "What is retrieval-augmented generation?";
    let embedding = embedder.embed(text).await?;
    println!("Embedded: \"{}\"", text);
    println!("Dimension: {}", embedding.len());
    println!(
        "First 5 values: {:?}\n",
        &embedding[..5.min(embedding.len())]
    );

    // Example 2: Batch embeddings
    println!("--- Example 2: Batch Embeddings ---");
    let texts = vec![
        "RAG combines retrieval and generation.",
        "It enables citation-aware responses.",
        "Local inference requires no API keys.",
    ];
    let embeddings = embedder
        .embed_batch(&texts.iter().map(|s| *s).collect::<Vec<_>>())
        .await?;
    println!("Embedded {} texts", embeddings.len());
    for (i, emb) in embeddings.iter().enumerate() {
        println!("  [{}] {} dimensions", i + 1, emb.len());
    }
    println!();

    // Example 3: Concurrent batch embeddings
    println!("--- Example 3: Concurrent Batch ---");
    let many_texts: Vec<&str> = (0..8)
        .map(|i| match i {
            0 => "Machine learning",
            1 => "Natural language processing",
            2 => "Deep learning",
            3 => "Transformers",
            4 => "Large language models",
            5 => "Retrieval systems",
            6 => "Vector databases",
            7 => "Knowledge graphs",
            _ => "Unknown",
        })
        .collect();

    let concurrent_embeddings = embedder.embed_batch_concurrent(&many_texts, 2).await?;
    println!(
        "Concurrently embedded {} texts (max_concurrent=2)",
        concurrent_embeddings.len()
    );
    println!(
        "All embeddings have dimension: {}\n",
        concurrent_embeddings[0].len()
    );

    // Example 4: Text completion
    println!("--- Example 4: Text Completion ---");
    let prompt = "Machine learning is";
    let completion = llm.complete(prompt).await?;
    println!("Prompt: \"{}\"", prompt);
    println!("Completion: \"{}\"\n", completion);

    // Example 5: Batch completions
    println!("--- Example 5: Batch Completions ---");
    let prompts = vec!["AI stands for", "GPU means", "Python is"];
    let completions = llm.complete_batch(&prompts).await?;
    println!("Generated {} completions:", completions.len());
    for (prompt, completion) in prompts.iter().zip(completions.iter()) {
        println!(
            "  \"{}...\" → \"{}\"",
            prompt,
            completion.chars().take(20).collect::<String>()
        );
    }
    println!();

    // Example 6: Health checks
    println!("--- Example 6: Health Checks ---");
    let embedder_healthy = embedder.health_check().await?;
    let llm_healthy = llm.health_check().await?;
    println!(
        "Embedder health: {}",
        if embedder_healthy {
            "✓ OK"
        } else {
            "✗ FAILED"
        }
    );
    println!(
        "LLM health: {}",
        if llm_healthy { "✓ OK" } else { "✗ FAILED" }
    );
    println!();

    // Example 7: Token estimation
    println!("--- Example 7: Token Estimation ---");
    let text_to_estimate = "The quick brown fox jumps over the lazy dog. This is a test sentence for token estimation.";
    let token_count = llm.estimate_tokens(text_to_estimate).await?;
    println!("Text: \"{}\"", text_to_estimate);
    println!("Estimated tokens: {}\n", token_count);

    // Example 8: Model info
    println!("--- Example 8: Model Information ---");
    let info = llm.model_info().await;
    println!("Model name: {}", info.name);
    println!("Version: {}", info.version.unwrap_or_default());
    println!(
        "Max context: {} tokens",
        info.max_context_length.unwrap_or(0)
    );
    println!("Supports streaming: {}", info.supports_streaming);
    println!();

    // Summary
    println!("=== Demo Complete ===");
    println!("✓ All async adapters working correctly");
    println!("✓ Ready for integration with oxidizedRAG");
    println!("\nNext steps:");
    println!("  1. Use LlamaAsyncEmbedder with oxidizedRAG's vector store");
    println!("  2. Use LlamaAsyncLanguageModel for RAG generation");
    println!("  3. Build knowledge graphs and retrieval pipelines");

    Ok(())
}
