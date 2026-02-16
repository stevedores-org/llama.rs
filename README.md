# llama.rs

High-performance LLM inference on Apple Silicon using Rust and MLX.

## Overview

This project provides a complete pipeline for running Llama-family models locally on Apple Silicon, leveraging MLX's Unified Memory Architecture for zero-copy GPU inference.

## Architecture

- **MLX Bindings** (`mlx/`): Safe Rust FFI wrappers around the MLX C API with RAII lifecycle management, lazy evaluation semantics, and `Send`/`Sync` thread safety.
- **Model** (`model/`): Full Llama 3 transformer architecture — Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), SwiGLU FFN, RMSNorm, and quantized linear layers.
- **KV Cache** (`cache/`): Pre-allocated key-value cache with O(1) `slice_update` writes — no allocation during the decode loop, no Metal kernel recompilation from shape changes.
- **Weights** (`weights/`): Memory-mapped safetensors loading with zero-copy MLX array creation and support for 4-bit/8-bit quantized weights.
- **Sampling** (`sampling/`): Temperature scaling, Top-K (via `argpartition`), Top-P (nucleus), and greedy — all implemented on GPU via MLX primitives.
- **Engine** (`engine/`): Two-phase inference (prefill + decode) with actor-based concurrency for non-blocking UI integration.
- **Tokenizer** (`tokenizer/`): BPE tokenizer with Llama 3 chat template formatting.

## Getting Started

```bash
cargo build
cargo run --bin llama-cli -- --help
```

### CLI Commands

```bash
# Show model architecture details
cargo run --bin llama-cli -- info 8b

# Inspect a model directory
cargo run --bin llama-cli -- inspect /path/to/model

# Interactive chat (requires Apple Silicon + MLX)
cargo run --bin llama-cli -- chat /path/to/model --temperature 0.7
```

### Running Tests

```bash
cargo test
```

## Features

- [x] MLX FFI bindings with safe Rust wrappers
- [x] Llama 3 transformer architecture (GQA, RoPE, SwiGLU)
- [x] Pre-allocated KV cache (O(1) updates)
- [x] Memory-mapped safetensors loading (zero-copy)
- [x] Quantized inference (4-bit/8-bit)
- [x] Sampling strategies (temperature, top-k, top-p, greedy)
- [x] Two-phase inference engine (prefill + decode)
- [x] Actor-based concurrency with token streaming
- [x] BPE tokenizer with chat templates
- [x] Interactive CLI

## Requirements

- Rust 1.70+
- Apple Silicon Mac (M1/M2/M3/M4) for GPU inference
- MLX C library installed (for macOS Metal acceleration)

On non-macOS platforms, the crate compiles with stub FFI implementations for development and testing.

## License

MIT
