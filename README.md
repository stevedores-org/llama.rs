# llama.rs

Rust bindings and tools for [llama.cpp](https://github.com/ggml-org/llama.cpp) - efficient LLM inference in Rust.

## Overview

This project provides Rust bindings and utilities for running large language models using the llama.cpp library, enabling fast and efficient inference on various hardware architectures.

## Planning Docs

- `docs/ROADMAP.md` - Epic + user stories + milestones (tracks GitHub issues #1 and #2)
- `docs/ARCHITECTURE.md` - target modular architecture (crates/modules, boundaries, invariants)
- `docs/TEST_STRATEGY.md` - TDD plan: unit/property/integration/perf testing and fixtures

## Getting Started

```bash
cargo build
cargo run --bin llama-cli
```

## Features

- [ ] Rust FFI bindings to llama.cpp
- [ ] High-level model loading API
- [ ] Inference session management
- [ ] Token streaming support
- [ ] Batch inference
- [ ] Memory-efficient inference

## License

MIT
