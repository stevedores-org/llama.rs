# Architectural & Performance Suggestions

This document outlines recommended future directions for `llama.rs`, focusing on performance, scalability, and robust RAG integration.

## 1. RAG Integration & Performance

### Embedding API Refinement
- **Batching**: The `LlamaEngine::embed` API should support explicit batching to leverage GPU (Metal) throughput.
- **Asynchronous Embeddings**: Retreival typically happens while the user is typing or during the prefill phase of a multi-turn conversation. Moving `embed` to an `async` trait will allow non-blocking retrieval.
- **Dimensionality Control**: Future models (like Matryoshka embeddings) allow variable output dimensions. The API should support requesting specific embedding sizes to optimize storage in `oxidizedRAG`.

### Citation Stability
- **KV Cache Pinning**: When generating answers based on retrieved context, the context tokens should be "pinned" in the KV cache to avoid re-computation if the generation is interrupted or branched.
- **Structured Output**: Use constrained sampling (similar to `llama-sampling` but with grammar/schema) to ensure citations (e.g., `[1]`, `[2]`) are correctly formatted and point to valid retrieved chunks.

## 2. Engine & Runtime Architecture

### Transition to Async Trait
- Current `LlamaEngine` is synchronous. For production use cases (especially `llama-server`), moving to `async trait` (or `RPITIT` in Rust 1.75+) is essential for:
    - Handling multiple concurrent sessions without thread-per-session.
    - Non-blocking I/O for weight loading and network-based retrieval.

### Unified Memory Management
- **Metal/CPU Shared Memory**: On Apple Silicon, we should use `MTLBuffer` with `Shared` storage mode to avoid copying tensors between CPU and GPU.
- **Paged KV Cache**: Implement a paged KV cache (vLLM style) in `llama-kv` early to support long-context windows and efficient memory usage across concurrent requests.

### Model Loading
- **Memory Mapping**: Use `mmap` for loading `.safetensors`. This allows the OS to handle paging and keeps the memory footprint low when multiple processes/threads access the same weights.
- **Lazy Loading**: Only load model blocks into GPU memory as they are needed during the first forward pass.

## 3. Sampling & Generation

### Speculative Decoding
- Implement a "draft" model (tiny Llama) to predict tokens, which are then verified by the main "target" model. This can significantly improve `tok/s` in high-latency backends.

### Grammar-Constrained Sampling
- Integrate a GBNF-style grammar engine into `llama-sampling` to allow the engine to guarantee JSON or Citations-compliant output.

## 4. Observability

### Structured Events
- Move beyond simple logging. Use the `tracing` crate to emit structured events for:
    - `TTFT` (Time To First Token)
    - `TPS` (Tokens Per Second)
    - `CacheHitRate` (for RAG context and KV cache)
    - `MetalPowerUsage` (on macOS)
