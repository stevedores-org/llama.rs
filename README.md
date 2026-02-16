# llama.rs

A modular Rust inference runtime for Llama-family models. Built on [oxidizedMLX](https://github.com/stevedores-org/oxidizedMLX) for Metal/CPU acceleration, with integrations for [oxidizedRAG](https://github.com/stevedores-org/oxidizedRAG) and [oxidizedgraph](https://github.com/stevedores-org/oxidizedgraph).

## Architecture

llama.rs uses a **"narrow waist"** design: the `llama-engine` crate defines the core `LlamaEngine` trait that all other crates depend on. Implementations can swap CPU/Metal/FFI backends without changing application code.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.

## Workspace Crates

| Crate | Description |
|-------|-------------|
| [`llama-engine`](crates/llama-engine) | Narrow-waist engine trait and core types |
| [`llama-tokenizer`](crates/llama-tokenizer) | Deterministic text-to-token conversion |
| [`llama-models`](crates/llama-models) | Model architectures (Llama/Qwen/Mistral) |
| [`llama-runtime`](crates/llama-runtime) | Backend selection and execution (oxidizedMLX) |
| [`llama-sampling`](crates/llama-sampling) | Sampling strategies (temperature, top-k/p) |
| [`llama-kv`](crates/llama-kv) | KV cache management and paging |

## Development

```bash
# Install just (task runner)
cargo install just

# Run all checks
just ci

# Individual commands
just fmt        # format code
just clippy     # lint
just test       # run tests
just check      # type-check
```

## License

MIT
