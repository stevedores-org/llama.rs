# Code Review: PR #18 — Workspace Restructure + Milestone A Scaffold

## Summary

This PR converts the single-crate project into a Cargo workspace with 6 crates (`llama-engine`, `llama-tokenizer`, `llama-models`, `llama-runtime`, `llama-sampling`, `llama-kv`), adds CI, a justfile, architecture docs, and provides real implementations for the tokenizer and KV cache crates.

**Build status:** Compiles clean, clippy passes with `-D warnings`, all 16 tests pass (9 in llama-kv, 7 in llama-tokenizer).

---

## Strengths

- **Clean commit history.** The 3 commits are well-scoped: scaffold → tokenizer → KV cache. Easy to review incrementally.
- **"Narrow waist" architecture is sound.** Having `LlamaEngine` as the central trait that downstream consumers (oxidizedRAG, oxidizedgraph) depend on is a good design. It enables backend swapping without breaking application code.
- **KV cache implementation is solid.** `LayerKVCache` and `SessionKVCache` correctly handle prefill/decode phases, validate shapes, and have thorough test coverage for edge cases (capacity exceeded, shape mismatch, clear, memory calculations).
- **CI is well-structured.** Separate jobs for fmt, clippy, and test (with matrix for ubuntu/macos) is the right setup for a Rust workspace.
- **Workspace dependency management.** Using `[workspace.package]` and `[workspace.dependencies]` for shared metadata avoids duplication.

---

## Issues

### Bug: `WhitespaceTokenizer::encode` ignores `self` state (Severity: Medium)

`crates/llama-tokenizer/src/lib.rs:111-120` — The `Tokenizer::encode` implementation creates a *new* `WhitespaceTokenizer` internally and builds vocab on it, then discards it. The actual token IDs returned are just positional indices (`enumerate → i as i32`), completely ignoring the vocabulary mapping that was just built:

```rust
fn encode(&self, text: &str) -> TokenizerResult<Vec<i32>> {
    let mut tokenizer = WhitespaceTokenizer::new();  // new instance, discards self
    tokenizer.ensure_vocab(text);

    Ok(text
        .split_whitespace()
        .enumerate()
        .map(|(i, _)| i as i32)  // always 0, 1, 2, ... regardless of vocab
        .collect())
}
```

This means `encode("hello world")` always returns `[0, 1]` and `encode("world hello")` also returns `[0, 1]` — "hello" and "world" don't have stable token IDs. The `decode` method has a similar issue: it fabricates `word_N` placeholders instead of using actual vocabulary.

**Recommendation:** Either make `WhitespaceTokenizer` properly bidirectional (encode assigns stable IDs from the shared vocab, decode uses the same vocab), or document it explicitly as a "positional index tokenizer for testing only" with no roundtrip guarantee. The `decode_roundtrip` test currently only checks that the *count* of words matches, which masks this bug.

### Bug: `KVLayout` enum is unused (Severity: Low)

`crates/llama-kv/src/lib.rs` — `LayerKVCache` stores a `layout: KVLayout` field, and the enum has three variants (`BySequence`, `ByHead`, `Transposed`), but `append_token` and `write_prefill` always use the same `BySequence` indexing (`offset = seq_len * n_heads * head_dim`). The layout is never consulted. This is fine for an initial scaffold, but the stored field creates a misleading API — callers might expect `KVLayout::ByHead` to actually change memory layout.

**Recommendation:** Either remove the `layout` field for now (add it when implemented), or add a doc comment noting it's reserved for future use.

### Design: `LlamaEngine` trait methods take `&self` but tokenize/detokenize are stateless (Severity: Low)

`crates/llama-engine/src/lib.rs:41-59` — `tokenize` and `detokenize` on the `LlamaEngine` trait require `&self`, but they don't conceptually need access to engine state. In practice they'll need a tokenizer loaded from the model, so `&self` is probably fine, but consider whether these should live on a separate `Tokenizer`-like trait (which already exists in `llama-tokenizer`). There's currently no dependency between `llama-engine` and `llama-tokenizer`, so the duplication of tokenization responsibility is worth thinking about.

### Design: `KVError` doesn't use `thiserror` (Severity: Nit)

`llama-kv` doesn't depend on `thiserror` and implements `Display` and `Error` manually, while `llama-engine` uses `thiserror`. For consistency across the workspace, consider adding `thiserror` to `llama-kv` as well, or document the choice.

### Nit: `clear()` doesn't zero the buffers

`LayerKVCache::clear()` resets `seq_len` to 0 but leaves stale data in the `k` and `v` vectors. This is fine for performance (the stale data is behind the `seq_len` cursor), but worth a doc comment to clarify the semantics — especially since stale data could leak if someone accesses the raw `Vec<f32>` fields directly (they're `pub`).

### Nit: Hardcoded `vocab_size` in `WhitespaceTokenizer`

`crates/llama-tokenizer/src/lib.rs:150` — `vocab_size()` returns `1000` as a hardcoded placeholder, which doesn't reflect the actual vocabulary state. Since `WhitespaceTokenizer` builds vocab dynamically, this could return `self.vocab.len()` instead.

---

## Minor Observations

- The `description` field was removed from the workspace `Cargo.toml` but added per-crate. Clean approach.
- `rust-toolchain.toml` pinning to `stable` is appropriate for a project at this stage.
- Three stub crates (`llama-models`, `llama-runtime`, `llama-sampling`) have `#![allow(unused)]` which is fine for scaffolding, but should be removed as they get populated.
- No inter-crate dependencies yet (e.g., `llama-engine` doesn't depend on `llama-tokenizer` or `llama-kv`). This is intentional per the architecture, but the wiring will need careful thought in the next milestone.

---

## Verdict

Good foundational PR. The architecture is well-considered, the KV cache implementation is production-quality for its scope, and the project infrastructure (CI, justfile, toolchain) is set up correctly. The main actionable item is the `WhitespaceTokenizer` encode/decode correctness — it should either work as a proper bidirectional tokenizer or be clearly documented as test-only with no roundtrip fidelity.
