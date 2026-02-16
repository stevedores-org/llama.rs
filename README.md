# llama.rs

[![Crates.io](https://img.shields.io/crates/v/llama-rs)](https://crates.io/crates/llama-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Rust bindings and tools for [llama.cpp](https://github.com/ggml-org/llama.cpp) - efficient LLM inference in Rust.

Published by [community-stevedores-org](https://crates.io/users/community-stevedores-org) on crates.io.

## Overview

This project provides Rust bindings and utilities for running large language models using the llama.cpp library, enabling fast and efficient inference on various hardware architectures.

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

## Organization

This crate is maintained by the [stevedores.org](https://crates.io/users/community-stevedores-org) organization. Other crates by this organization include:

- [graphrag-cli](https://crates.io/crates/graphrag-cli) - Modern TUI for GraphRAG operations
- [graphrag-core](https://crates.io/crates/graphrag-core) - Core portable library for GraphRAG
- [graphrag-server](https://crates.io/crates/graphrag-server) - REST API server for GraphRAG
- [mlx-core](https://crates.io/crates/mlx-core) - Safe Rust API for MLX tensors, devices, and core operations
- [mlx-cpu](https://crates.io/crates/mlx-cpu) - Pure Rust CPU backend for MLX
- [mlx-io](https://crates.io/crates/mlx-io) - Tensor serialization: safetensors, GGUF, mmap loading
- [mlx-ops](https://crates.io/crates/mlx-ops) - Op registry, broadcasting rules, and dtype promotion for MLX
- [mlx-roadmap](https://crates.io/crates/mlx-roadmap) - MLX development roadmap and planning tools
- [nix-env-manager](https://crates.io/crates/nix-env-manager) - Nix Flakes and Attic cache integration for AIVCS
- [oxidized-state](https://crates.io/crates/oxidized-state) - SurrealDB backend for AIVCS state persistence

## License

MIT
