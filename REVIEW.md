## Code Review: PR #44 — feat: LLAMA-004 paging/eviction interface for KV cache

### Summary

This PR adds sliding-window KV cache eviction: a `KvPager` trait, `evict_prefix()` on `LayerKVCache`, and `KvPager` implementations for both `LayerKVCache` and `SessionKVCache`. The implementation is correct for all three layouts, tests pass, and clippy is clean. There are a few issues worth addressing.

---

### Bug / Correctness

**1. `SessionKVCache::apply_sliding_window` can leave layers in an inconsistent state**
(`lib.rs` lines 335–348 in the PR)

The method computes the eviction count from the first layer, then iterates through all layers calling `evict_prefix`. If any layer has a different `seq_len` (violating the assumed invariant), the eviction could succeed on earlier layers and then fail on a later one, leaving the session partially evicted.

A defensive fix: validate all layers *before* mutating any:

```rust
fn apply_sliding_window(&mut self, window: usize) -> Result<usize, KVError> {
    let evict = self
        .layers
        .first()
        .map(|l| l.seq_len.saturating_sub(window))
        .unwrap_or(0);
    // Validate before mutating
    for layer in &self.layers {
        if evict > layer.seq_len {
            return Err(KVError::EvictionOutOfRange {
                requested: evict,
                available: layer.seq_len,
            });
        }
    }
    for layer in &mut self.layers {
        layer.evict_prefix(evict)?;
    }
    Ok(evict)
}
```

This is a low-probability issue in practice (since layers should always have matching `seq_len`), but it's worth guarding against because a partially-mutated session cache would be very hard to debug.

---

### Nit / Cleanup

**2. Stale module-level doc comment**
(`lib.rs` line 9)

The crate doc still says `"Future: paging/eviction, sliding window"`. Since this PR implements that functionality, update it:

```rust
//! - Paging/eviction, sliding window
```

---

### Testing Gaps

**3. No eviction tests for `ByHead` or `Transposed` layouts**

All four new tests use `KVLayout::BySequence`. The `evict_prefix` implementation relies on layout-aware `read_token_at_position`/`write_token_at_position`, which should handle all layouts correctly, but there are no tests confirming this. Adding at least one eviction test per alternate layout would catch any indexing bugs there.

**4. No tests for eviction boundary cases `count == 0` and `count == seq_len`**

`evict_prefix` has explicit early-return paths for both — `count == 0` (noop) and `count == seq_len` (delegates to `clear()`). Neither is tested.

**5. V tensor values not verified after eviction**

`evict_prefix_compacts_oldest_tokens` only asserts on `cache.k` values. Adding assertions on `cache.v` would verify both tensors are compacted correctly.

---

### Performance (non-blocking observation)

**6. `evict_prefix` allocates temporaries on every call**

The method allocates `k_tmp` and `v_tmp` vectors each invocation. For hot-path usage (e.g., every decode step in a streaming scenario), this will generate allocation pressure. For `BySequence` layout, the compaction could be a single `copy_within` on the backing slices with no allocation. For other layouts, a reusable scratch buffer stored on the struct would help.

Not a blocker for merging — just worth noting for future optimization when this lands on a hot path.

---

### Verdict

The core logic is correct and well-structured. I'd recommend addressing **item 1** (the partial-mutation risk in `SessionKVCache`) before merging, and **items 3–5** (test gaps) as a follow-up if not addressed here. The rest is non-blocking.
