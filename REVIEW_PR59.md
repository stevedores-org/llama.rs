# Code Review: PR #59 — Merge develop into main

## Summary

This is a large integration PR consolidating 33 commits from `develop` into `main`. It adds: KV cache sliding-window eviction and atomic session operations, a sampling crate rewrite, backend selection with feature gates, inference telemetry, MockEngine for the narrow-waist API, backend parity gates, and ~122 new tests across all crates.

**Build**: ✅ Clean (`cargo build --workspace`)
**Tests**: ✅ 122/122 pass (`cargo test --workspace`)
**Clippy**: ✅ No warnings

---

## Strengths

1. **Excellent test coverage** — 122 tests across 6 test suites with thoughtful property tests, boundary conditions, bug-injection tests (off-by-one detection), and concurrent stress tests.

2. **Atomic session operations** — `SessionKVCache::append_token` and `write_prefill` pre-validate before mutating, maintaining the synchronized seq_len invariant across layers. The rollback behavior in `write_prefill` (clear on error) is a solid defensive choice.

3. **Clean API redesign in llama-sampling** — The rewrite replaces the builder pattern with a validated `SamplingConfig` struct and up-front validation in `Sampler::new()`. The `thiserror`-based errors with descriptive variants are a clear improvement.

4. **KvPager trait** — Nice abstraction for sliding-window behavior with layout-aware eviction across BySequence, ByHead, and Transposed layouts.

5. **Backend abstraction** — Feature-gated `Backend` enum with `KernelMatrix` probing and `BackendSelector` provides a clean extension point for Metal support.

---

## Issues

### 1. [Medium] `evict_prefix` uses O(n) temporary allocation per token

`LayerKVCache::evict_prefix` at `crates/llama-kv/src/lib.rs` allocates a `k_tmp`/`v_tmp` buffer and copies token-by-token through `read_token_at_position` + `write_token_at_position`. For the `BySequence` layout, this could be a single `copy_within` on the underlying slice, which would be significantly faster and avoid the allocation entirely.

```rust
// Current: O(keep) allocations + 2 copies per token
let mut k_tmp = vec![0.0f32; token_width];
let mut v_tmp = vec![0.0f32; token_width];
for new_seq in 0..keep {
    let old_seq = new_seq + count;
    self.read_token_at_position(old_seq, &mut k_tmp, &mut v_tmp);
    self.write_token_at_position(new_seq, &k_tmp, &v_tmp);
}
```

The single allocation is fine, but the token-by-token copy through the index function is unnecessarily expensive for `BySequence` layout where data is contiguous. Consider a layout-aware fast path using `slice::copy_within`.

### 2. [Medium] `write_prefill` rollback semantics destroy valid state

`SessionKVCache::write_prefill` at `crates/llama-kv/src/lib.rs` calls `self.clear()` on any layer failure. If layer 0 succeeds but layer 1 fails, layer 0's valid data is destroyed. The docstring says "rollback on error" but it's actually "destroy everything on error." This is acceptable if documented clearly — but the name "rollback" implies restoring the previous state, which it does not do. Either:
- Rename/re-document to "clear on error" semantics, or
- Save `seq_len` values before mutation and restore them on error (true rollback)

Since `write_prefill` requires `seq_len == 0` (the `NotEmpty` check in `LayerKVCache::write_prefill`), clearing is effectively a true rollback in practice. But if that invariant ever changes, this will silently become destructive.

### 3. [Medium] `attention_single_head_metal` is just `attention_single_head_cpu`

`crates/llama-runtime/src/lib.rs` defines `attention_single_head_metal` which simply calls `attention_single_head_cpu`. The parity tests then trivially pass since both backends execute identical code. This means the parity gate infrastructure (`BackendParityGate`, concurrent stress tests, etc.) is testing that `f(x) == f(x)`, not actual backend parity.

This is acceptable for a placeholder/scaffold, but should be clearly marked as such (e.g., `#[cfg(not(feature = "real_metal"))]` or a `TODO` comment). The current code and docs imply more than what's actually being tested.

### 4. [Low] `SessionKVCache::seq_len()` will panic on empty layers vec

After adding `assert!(n_layers > 0)` to `SessionKVCache::new`, `seq_len()` changed from `self.layers.first().map(...).unwrap_or(0)` to `self.layers[0].seq_len`. The panic guard in `new` protects construction, but if `SessionKVCache` were ever constructed via deserialization or unsafe code with an empty `layers` vec, `seq_len()` would panic instead of returning 0. The assert in `new` makes this unlikely in practice but worth noting.

### 5. [Low] `Sampler::sample` returns `i32` but token IDs are conceptually `u32`

The new sampling API returns `i32` while checking `history` for negative values (returning `InvalidHistoryToken`). This suggests token IDs should be non-negative, but the type system doesn't enforce it. The `LlamaEngine` trait uses `TokenId` which may be unsigned. Consider using the same `TokenId` type throughout for consistency.

### 6. [Low] `apply_repetition_penalty` marks seen tokens but doesn't deduplicate penalty

The function uses a `seen` vec to skip duplicate history entries, which is correct. But the `seen` vec is allocated as `vec![false; logits.len()]` — for large vocabularies (32k+), this is a 32KB allocation on every sample call. A `HashSet` over the (typically small) history would be more efficient for real-world vocab sizes.

### 7. [Nit] `SessionKVCache::layers` field access in tests

Tests like `session_sliding_window_applies_to_all_layers` directly access `session.layers` despite the field being made private. This only works because the test module is inside the same crate. Integration tests won't be able to do this. Consider adding a `layers()` accessor or iterating via the `layer(idx)` accessor in tests for API consistency.

### 8. [Nit] Redundant `Result` qualification

Several functions in `lib.rs` use `std::result::Result<T, RuntimeError>` instead of importing or defining a type alias. This was likely done to avoid conflicts with the `llama_engine::Result` import, but a local `type Result<T> = std::result::Result<T, RuntimeError>;` would be cleaner.

---

## Suggestions (not blocking)

- Consider splitting this mega-PR into logical sub-PRs (KV eviction, sampling rewrite, backend abstraction, telemetry, MockEngine) for easier review and bisection. The 33-commit merge makes it hard to review individual changes.
- The `ToyModel` in runtime is growing (now ~100 lines with seeded jitter). Consider extracting it to a `test_utils` module.
- `tests/README.md` references directories (`tokenizer/`, `sampler/`, `inference/`, etc.) that don't exist yet. The actual test files are in crate-level `tests/` dirs. Update the README to match reality or add a note that this is aspirational.

---

## Verdict

The code is well-structured, all tests pass, and clippy is clean. The main concerns are around the placeholder Metal backend creating a false sense of parity validation, and the `write_prefill` "rollback" semantics. Neither is blocking for a develop→main merge, but should be tracked as follow-up items.

**Recommendation: Approve with comments** — merge is safe, but address items 1-3 in follow-up PRs.
