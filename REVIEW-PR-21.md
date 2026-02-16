# Code Review: PR #21 — Sampling Suite

**PR:** feat: Sampling suite - greedy, top-k/p, temperature, repetition penalty
**Author:** stevei101
**Base:** develop ← feat/milestone-a-sampling
**Reviewer:** Claude (automated)
**Verdict:** Request Changes

---

## Summary

This PR introduces sampling strategies (greedy, top-k/p, temperature, repetition penalty)
for the `llama-sampling` crate and makes structural changes to `llama-engine`. However,
the PR branch (`feat/milestone-a-sampling`) forked from an early commit (`1f86ecd`) and
**has not been rebased onto current `develop`**. Since then, `develop` has already received
a more complete sampling implementation (PR #29, commit `2cd1e99`) and engine struct
improvements (`df6a05c`).

**As written, merging this PR would regress several improvements already on `develop`.**

---

## Critical Issues

### 1. RNG state is never advanced between `sample()` calls (Bug)
**File:** `crates/llama-sampling/src/lib.rs` — `Sampler::sample()`

`sample()` takes `&self` (immutable) and creates a brand-new `SeededRng::new(self.seed)`
on every invocation. This means **every call to `sample()` produces the same token**,
making the sampler useless for multi-token generation.

```rust
// Current (broken):
pub fn sample(&self, logits: &[f32]) -> SamplingResult<usize> {
    // ...
    let mut rng = SeededRng::new(self.seed); // re-created every call
    self.sample_from_distribution(&probs, &mut rng)
}
```

**Fix:** The `develop` branch already fixes this by storing `rng: SeededRng` as a field
and making `sample()` take `&mut self`, so the RNG state persists and advances across calls.

### 2. Repetition penalty is wrong for negative logits (Bug)
**File:** `crates/llama-sampling/src/lib.rs` — `Sampler::sample_with_history()`

All logits are divided by the penalty regardless of sign:
```rust
work_logits[token_id] /= penalty;
```

For negative logits (e.g., `-2.0`), dividing by a penalty > 1 makes the value *less*
negative (`-2.0 / 2.0 = -1.0`), which **increases** the token's probability — the exact
opposite of the intended effect.

**Fix:** Divide positive logits by penalty, multiply negative logits by penalty. The
`develop` branch already has this correct:
```rust
if work_logits[token_id] > 0.0 {
    work_logits[token_id] /= penalty;
} else {
    work_logits[token_id] *= penalty;
}
```

### 3. `apply_top_k` panics on `k=0` (Bug)
**File:** `crates/llama-sampling/src/lib.rs` — `Sampler::apply_top_k()`

```rust
let threshold = indexed[k - 1].1; // panics if k == 0 (underflow to usize::MAX)
```

No guard for `k == 0` or `k >= logits.len()`. The `develop` branch guards both cases.

---

## Design/Structural Issues

### 4. Branch is stale — would regress `develop`
The merge base is `1f86ecd` (initial workspace scaffold). Since then, `develop` has received:
- PR #29: Complete sampling implementation with correct RNG, repetition penalty, xorshift64
- `df6a05c`: `Session::id` made private with proper `Uuid` type, getter method
- Multiple engine struct improvements

Merging this PR would overwrite all of these improvements.

### 5. `Session.id` downgraded from `Uuid` to `String` (Regression)
**File:** `crates/llama-engine/src/lib.rs`

The PR changes `Session.id` from `uuid::Uuid` (type-safe, 128-bit) to `String`
(heap-allocated, no type safety) and makes the field `pub`. The `develop` branch
intentionally keeps `id` as `uuid::Uuid` with a private field and `id()` getter to
preserve invariants.

### 6. `Session` made `Clone` (Regression)
The `develop` branch explicitly documents that `Session` is not `Clone` because cloning
would imply duplicating KV cache state. This PR adds `#[derive(Clone)]`.

### 7. `DecodeResult` renamed to `TokenStream` (Misleading)
`TokenStream` implies a stream/iterator abstraction, but the struct holds a single token.
`develop` uses `DecodeResult`, which accurately describes what it is.

### 8. `embed()` method removed from `LlamaEngine` trait
The `develop` branch includes `fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>`
for oxidizedRAG integration. This PR removes it.

### 9. `#[must_use]` removed from `PrefillResult`
Dropping a `PrefillResult` silently discards important information (tokens_processed count).

### 10. Duplicate trait documentation
The `LlamaEngine` trait has two doc blocks stacked on top of each other — the old comment
and a new one. One should be removed.

---

## Minor Issues

### 11. Unused `HashMap` import
```rust
use std::collections::HashMap; // never used
```

### 12. Redundant `next_uniform()` method
`SeededRng::next_uniform()` is identical to `next_f32()`. One should be removed.

### 13. Weaker RNG algorithm
The PR uses a simple LCG (`state * 1103515245 + 12345`), which has known poor statistical
properties (low bits have short periods). The `develop` branch uses xorshift64, which is
better suited for sampling. The PR also doesn't handle `seed=0` (LCG with state 0 can
degenerate).

### 14. Test coverage gaps
- `deterministic_generation` test only calls `sample()` once per sampler, which always
  passes even with the RNG bug. It should test *sequential* calls.
- No test for `rng_advances_between_calls` (the key test that would catch issue #1).
- No test for negative-logit repetition penalty behavior.
- `seeded_rng_reproducible` only tests 10 iterations (develop tests 100 + range checks).

---

## Recommendation

**Request changes.** This PR should not be merged in its current form. The branch needs
to be rebased onto current `develop` and reconciled with the existing sampling
implementation. The critical RNG state bug and repetition penalty logic error must be
fixed regardless.

Options:
1. **Close this PR** — The work has already been superseded by PR #29 and subsequent
   improvements on `develop`.
2. **Rebase and reconcile** — If there are features here not covered by `develop`, rebase
   onto `develop` and submit only the delta.
