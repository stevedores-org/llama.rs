# Changelog

All notable changes to llama.rs are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is
pre-1.0 and does not yet emit semver-stable releases.

Each entry is tagged with the PR that introduced it.

## [Unreleased]

### Changed (breaking output)

- **`llama-tokenizer`: `WhitespaceTokenizer` now preserves whitespace as its
  own tokens** *(via #92)*. Previously, `split_whitespace()` dropped leading,
  trailing, and interior whitespace runs, so `"hello world"` encoded to 2
  token ids and `"  a  "` round-tripped to `"a"`. The new encoder treats each
  run of whitespace and each run of non-whitespace as a separate piece, so
  `"hello world"` encodes to 3 ids and round-trip is bit-perfect.

  Any caller that hard-codes token counts or persists stored token-id
  sequences from pre-change versions will break. There is no compatibility
  shim — re-encode prompts and rebuild any artifacts that store token-id
  sequences.

### Changed

- **`llama-kv` / `llama-models`: `KVLayout` consolidated** *(via #97)*. The
  enum lives in `llama-kv` only; `llama-models` re-exports it as
  `llama_models::KVLayout`. The duplicated definition is gone, and the CLI
  no longer needs a translation match between the two.

- **`llama-models::attention_decode`: capacity contract clarified** *(via
  #92, documented in #97)*. `keys` and `values` must now be pre-allocated to
  full capacity (`max_seq_len * n_heads * head_dim`), not trimmed to the
  populated `seq_len * n_heads * head_dim`. This is required because the
  `ByHead` and `Transposed` layouts stride on `max_seq_len`. Callers passing
  a slice trimmed to `seq_len` will fail the shape check (regression test:
  `attention_decode_rejects_undersized_buffer`).

- **`llama-agents::orchestration`: multi-dependency input plumbing** *(landed
  on develop ahead of #97)*. `RoleExecutor` is now `fn(&[&Artifact]) ->
  Artifact`, receiving every dependency's output as a slice in
  `OrchestrationTask.depends_on` order. The previous signature silently
  collapsed multi-dep inputs to `depends_on[0]`. `inputs[0]` remains the
  primary handoff contract-validated against the node's `HandoffContract`;
  secondary deps are scheduling/data inputs and are not contract-validated.

### Fixed

- **KV layout bug** *(via #92)*: `attention_decode` is now layout-aware and
  reads K/V via the layout's indexing function instead of a fixed
  `BySequence`-only stride. Regression test:
  `crates/llama-cli/tests/layout_bug.rs`.
- **Tokenizer fidelity** *(via #92)*: round-trip is now bit-perfect for
  whitespace, tabs, newlines, multi-byte UTF-8, and emoji. See
  `crates/llama-tokenizer/tests/`.
- **`ToyModel`** *(via #92)*: generalized to arbitrary `dim` / `vocab_size`
  instead of hardcoded `dim=2`, `vocab=8`.
- **Workspace `Cargo.toml`** *(via #92)*: removed duplicate `crates/llama-rag`
  member entry.
