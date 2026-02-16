## Code Review: PR #32 — feat: LLAMA-003 foundational sampling crate

### Summary
This PR implements the `llama-sampling` crate with greedy, temperature, top-k, top-p, and repetition penalty sampling backed by a deterministic XorShift64 RNG. The sampling implementation is solid and well-structured. However, the PR bundles several unrelated destructive changes that should be separated.

---

### Scope Concern: Unrelated Changes Bundled In

**This is the most important issue with this PR.** The commit titled "feat(sampling): implement deterministic greedy/top-k/top-p/temperature + repetition penalty" includes changes well beyond the sampling crate:

1. **Deletion of the entire `llama-kv` implementation** (~365 lines removed, replaced with a 4-line stub). This is a **breaking regression** — `llama-kv` had a working `LayerKVCache`, `SessionKVCache`, `KVShape`, `KVLayout`, and `KVError` with full test coverage. All of that is gone.

2. **Stripping `ModelHandle` and `Session` in `llama-engine`** from structs with fields (`id: u64`) and derived traits to bare unit structs. This breaks any downstream code that references `.id` or relies on `Debug`/`Clone`/`Hash` derives.

3. **Deletion of `.github/LABELS.md` and `scripts/create-labels.sh`** — repo housekeeping files unrelated to sampling.

4. **README.md edit** removing the LABELS.md reference.

**Recommendation:** Split this into separate PRs:
- One PR for the sampling crate (the actual feature).
- One PR for the `llama-kv` gutting (with justification — is it being rewritten?).
- One PR for the `llama-engine` struct changes.
- One PR for the label/script cleanup.

Combining feature additions with destructive deletions in a single commit makes it impossible to accept the good work without also accepting regressions.

---

### Sampling Implementation Review (`crates/llama-sampling/src/lib.rs`)

The core sampling code is well-written. Specific feedback:

#### Correctness Issues

**1. `apply_top_p` — off-by-one in the keep logic**

```rust
for (rank, &(idx, p)) in order.iter().enumerate() {
    cumulative += p;
    keep[idx] = true;
    if cumulative >= top_p && rank > 0 {
        break;
    }
}
```

The `rank > 0` guard means that when the first token alone exceeds `top_p`, the loop continues to a second token unconditionally. This is likely intentional (always keep at least 2 candidates), but it is undocumented and surprising. If the intent is "always keep at least 1 token," the guard should be removed — the first token is already marked `keep[idx] = true` before the check. If the intent is "always keep at least 2," add a comment explaining why.

**2. `apply_repetition_penalty` treats out-of-range tokens as errors**

```rust
if idx >= logits.len() {
    return Err(SamplingError::InvalidHistoryToken(token));
}
```

In practice, token histories from prior context windows may legitimately contain token IDs beyond the current logits slice length (e.g., if logits are truncated or the vocabulary changes between model versions). Silently skipping out-of-range tokens with a `continue` would be more robust. At minimum, this design decision should be documented.

**3. `normalize_probs` silently no-ops on all-zero distributions**

```rust
if sum <= 0.0 {
    return;
}
```

If all probabilities are zero after top-k/top-p filtering, `normalize_probs` returns without modification, leaving an all-zero distribution. The downstream `sample_from_probs` handles this with an argmax fallback, but this creates a subtle contract: `normalize_probs` does NOT guarantee a valid distribution. This should be documented, or the function should return a `Result`.

#### Design Suggestions

**4. Token type — `i32` is unusual for token IDs**

Token IDs are inherently non-negative. Using `i32` means every function must validate against negative values (as `apply_repetition_penalty` does). A `u32` or a newtype like `TokenId(u32)` would eliminate an entire class of errors at the type level and remove the need for the `InvalidHistoryToken` variant for negative values.

**5. `SamplingConfig::validate` is private but `SamplingConfig` fields are public**

Users can construct an invalid `SamplingConfig` directly (e.g., `temperature: -1.0`) and it will only be caught at `Sampler::new()` time. Either:
- Make `validate` public so users can check early, or
- Use a builder pattern / make fields private with accessors to enforce invariants at construction.

**6. `XorShift64::next_f32` only uses 24 bits of entropy**

This is a reasonable approach for f32 mantissa precision, but worth noting it means the RNG can only produce ~16M distinct probability thresholds. For vocabulary sizes >16M this could introduce sampling bias. Not a problem today, but worth a comment.

**7. Missing `#[must_use]` on `greedy_sample` and `apply_repetition_penalty`**

Both are public functions returning `Result` — adding `#[must_use]` is idiomatic Rust and prevents accidentally ignoring errors.

#### Style / Minor

**8.** `SamplingConfig` derives `Clone` but not `PartialEq`. Adding `PartialEq` would be useful for test assertions and consistency with `SamplingError` which does derive it.

**9.** The `greedy_sample` function is `pub` but `softmax_with_temperature`, `normalize_probs`, `apply_top_k`, `apply_top_p`, and `sample_from_probs` are private. Consider whether `apply_top_k` and `apply_top_p` should also be public — they're useful building blocks for custom sampling pipelines.

**10.** The `seeded_rng_is_deterministic` test feeds `seq_a`/`seq_b` back as history, which means the test simultaneously validates determinism AND repetition penalty interaction. A cleaner test would pass empty history to isolate the RNG determinism concern, then have a separate test for penalty + determinism.

---

### Tests

The test suite covers the key scenarios well. Additional tests to consider:
- **Edge case:** single-element logits vector.
- **Edge case:** `top_k = 0` (currently a no-op, but is that intentional?).
- **Edge case:** `repetition_penalty = 1.0` (identity, should be a no-op).
- **NaN handling:** what happens if logits contain NaN? The current `partial_cmp` fallbacks to `Ordering::Equal`, which may produce surprising results.
- **Temperature near zero:** e.g., `temperature = 1e-30` — does softmax overflow?

---

### Verdict

The sampling implementation is good work and close to merge-ready **on its own**. However, this PR cannot be approved as-is because it deletes the `llama-kv` crate implementation and modifies `llama-engine` structs without justification. Please split the unrelated changes into separate PRs.

**Requested action:** Split into focused PRs and address the `apply_top_p` off-by-one documentation / intent clarification.
