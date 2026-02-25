# CODEX.md

Operating instructions for AI/code agents working in this repository.

## Scope

- Applies to the entire workspace.
- Make focused, minimal changes per PR.

## Branch and PR Rules

- Base branch is `develop`.
- Create feature branches from `develop`.
- Keep one change theme per PR.

## Read First

- `README.md`
- `CONTRIBUTING.md`
- `docs/ARCHITECTURE.md`
- `docs/TEST_STRATEGY.md`

## Module Ownership Map

- Engine contracts: `crates/llama-engine/src/lib.rs`
- Models/loading: `crates/llama-models/src/lib.rs`
- Runtime/backends: `crates/llama-runtime/src/*`
- Sampling: `crates/llama-sampling/src/lib.rs`
- Tokenization: `crates/llama-tokenizer/src/lib.rs`
- KV cache: `crates/llama-kv/src/lib.rs`
- Server/API: `crates/llama-server/src/*`
- CLI: `crates/llama-cli/src/*`
- RAG/agents: `crates/llama-rag/src/*`, `crates/llama-agents/src/lib.rs`

## Coding and Safety Expectations

- Preserve public behavior unless the PR explicitly changes contract.
- Prefer explicit error handling over hidden fallbacks.
- Avoid unrelated refactors and formatting churn.

## Validation

Run at minimum:

```bash
just test
```

Run full checks when feasible:

```bash
just fmt
just clippy
just ci
```

## PR Checklist

- Problem and approach are clearly described.
- Tests demonstrate behavior.
- Compatibility/risk impacts are called out.
- Changed files align with the module ownership map.
