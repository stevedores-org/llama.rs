# Contributing to llama.rs

This guide is a practical map for contributors who want to land changes quickly and safely.

## Local Setup

```bash
git clone https://github.com/stevedores-org/llama.rs.git
cd llama.rs

# task runner used by this repo
cargo install just

# baseline checks
just ci
```

## Start Contributing Map

### 1) Core engine contracts and shared types
- Primary files: `crates/llama-engine/src/lib.rs`
- Use this area for trait boundaries, generation interfaces, and shared runtime contracts.

### 2) Model loading and architecture behavior
- Primary files: `crates/llama-models/src/lib.rs`
- Tests: `crates/llama-models/tests/unaligned_safetensors.rs`
- Use this area for model graph correctness, tensor layout assumptions, and loader changes.

### 3) Runtime backend execution and telemetry
- Primary files: `crates/llama-runtime/src/lib.rs`, `crates/llama-runtime/src/backend.rs`, `crates/llama-runtime/src/telemetry.rs`
- Tests: `crates/llama-runtime/tests/kv_equivalence.rs`
- Use this area for backend routing, execution performance, and observability hooks.

### 4) Tokenization and sampling behavior
- Tokenizer: `crates/llama-tokenizer/src/lib.rs`, tests in `crates/llama-tokenizer/tests/roundtrip.rs`
- Sampling: `crates/llama-sampling/src/lib.rs`
- Use this area for deterministic token transforms and sampling policy changes.

### 5) KV cache semantics
- Primary files: `crates/llama-kv/src/lib.rs`
- Use this area for cache layout, paging, eviction, and cache correctness/perf tradeoffs.

### 6) API server and HTTP contracts
- Primary files: `crates/llama-server/src/main.rs`, `crates/llama-server/src/server.rs`, `crates/llama-server/src/handlers/*`, `crates/llama-server/src/models/*`
- Tests: `crates/llama-server/tests/integration_test.rs`
- Use this area for endpoint behavior, request/response compatibility, streaming responses, and session handling.

### 7) CLI and end-user flows
- Primary files: `crates/llama-cli/src/main.rs`, `crates/llama-cli/src/lib.rs`
- Use this area for UX, command behavior, and local operator workflows.

### 8) RAG and agent integrations
- RAG: `crates/llama-rag/src/*`
- Agents: `crates/llama-agents/src/lib.rs`
- Use this area for orchestration and external integration boundaries.

### 9) Docs and planning artifacts
- Architecture and roadmap: `docs/*.md`
- Root overview: `README.md`
- Use this area for contributor guidance, roadmap updates, and design clarity.

## Contribution Workflow

1. Branch from `develop`.
2. Keep each PR focused on one theme.
3. Add/update tests with behavior changes.
4. Run:

```bash
just fmt
just clippy
just test
```

5. Open PR to `develop` with:
- problem statement
- approach summary
- test evidence
- compatibility/risk notes

## PR Review Expectations

- Correctness first, then performance/ergonomics.
- No silent behavior changes in API paths.
- Explicitly call out compatibility impact.
- Deterministic tests for new behavior and edge cases.
