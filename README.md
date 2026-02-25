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

## Planning Docs

- [`docs/ROADMAP.md`](docs/ROADMAP.md) — Epic + user stories + milestones (tracks [#1](https://github.com/stevedores-org/llama.rs/issues/1), [#2](https://github.com/stevedores-org/llama.rs/issues/2))
- [`docs/PROGRESS_REPORT.md`](docs/PROGRESS_REPORT.md) — Progress: milestones A–E, Milestone A checklist with PRs, Epic 1–5 status
- [`docs/MILESTONE_A.md`](docs/MILESTONE_A.md) — Milestone A "Hello Inference" checklist and next (tiny model forward pass)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Modular architecture, crate boundaries, invariants
- [`docs/TEST_STRATEGY.md`](docs/TEST_STRATEGY.md) — TDD plan: unit/property/golden/parity/perf testing
- [`.github/LABELS.md`](.github/LABELS.md) — Recommended gh labels (`cursor`, `llama.rs`, `docs`, `roadmap`, `priority`, `rag`); create with `./scripts/create-labels.sh`

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

## Contributing

- Start here: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Agent/operator instructions: [`CODEX.md`](CODEX.md)

## License

MIT
