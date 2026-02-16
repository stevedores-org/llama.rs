# Code Review: PRs #29, #30, #31

**Reviewer:** Claude
**Date:** 2026-02-16
**All tests passing:** 42/42 (13 sampling + 16 KV cache + 4 runtime + 9 tokenizer)

---

## PR #29 — `llama-sampling`: Sampling Strategies

**Branch:** `claude/feat-sampling` | **Scope:** `crates/llama-sampling/src/lib.rs` (+446 lines) | **Tests:** 13/13 pass

### Summary

Implements `Sampler` with builder API: greedy (near-zero temp), temperature scaling, top-k, top-p nucleus, repetition penalty, seeded xorshift64 RNG. Also rewrites `llama-tokenizer` from stub into proper `Tokenizer` trait + `WhitespaceTokenizer`.

### Strengths

- Clean builder pattern — idiomatic, composable
- Repetition penalty correctly handles positive/negative logits (divide vs multiply)
- xorshift64 with zero-state guard is appropriate for reproducible test harness
- Good edge-case coverage: empty logits, invalid temperature, negative logits with penalty, determinism, distribution variance

### Issues

#### 1. `apply_top_k` tie-breaking (minor)
`lib.rs:163-172` — threshold comparison `< threshold` keeps **all** tokens at the boundary value. With logits `[3.0, 3.0, 3.0, 1.0]` and k=2, all three 3.0 values survive → top-3 instead of top-2. Common in implementations but should be documented.

#### 2. `apply_top_p` same tie behavior (minor)
`lib.rs:180-199` — keeps all tokens with `prob >= cutoff_prob`, which can exceed the cumulative threshold `p`. Worth a doc comment.

#### 3. Temperature check ordering (nit)
`<= 0.0` rejected at line 138, near-greedy short-circuit at line 152. Temperatures in `(0.0, 1e-3)` go through unnecessary softmax before hitting argmax. Moving the check earlier would be cleaner.

#### 4. `SeededRng::next_f32` bit usage (nit)
`>> 40` discards 16 usable bits. Consider `>> 32` for better distribution quality before f32 truncation.

### Verdict: **Approve with minor comments**

---

## PR #30 — `llama-kv`: KV Cache

**Branch:** `claude/feat-kv-cache` | **Scope:** `crates/llama-kv/src/lib.rs` (+547), `crates/llama-engine/src/lib.rs` | **Tests:** 16/16 pass

### Summary

Implements `LayerKVCache` (single-layer K/V storage, prefill + decode) and `SessionKVCache` (multi-layer orchestrator, synchronized seq_len). Adds `id: u64` to `ModelHandle`/`Session` in `llama-engine`.

### Strengths

- Clean layering: `LayerKVCache` → `SessionKVCache`
- `write_prefill` enforces empty-cache precondition explicitly
- Dual memory tracking: `memory_bytes()` (allocated) vs `active_memory_bytes()` (in-use)
- `KVLayout` enum preserves future flexibility without current complexity
- `thiserror` derive for clean error types

### Issues

#### 1. `write_prefill` remainder not validated (minor)
`lib.rs:186` — integer division `k_seq.len() / (n_heads * head_dim)` silently drops remainder. The subsequent `total_len` check catches mismatches, but the error message is confusing. A dedicated "not evenly divisible" check would be clearer.

#### 2. `append_token` is not atomic (documented, acceptable for Milestone A)
`lib.rs:261-271` — partial failure leaves layers in inconsistent state. Doc comment acknowledges this. Production version should pre-validate or roll back.

#### 3. `seq_len()` panics on zero-layer cache (minor)
`lib.rs:241-245` — `SessionKVCache::new(0, ...)` creates valid struct that panics on `seq_len()`. Validate `n_layers > 0` in constructor.

#### 4. `KVShape` unused in cache operations (nit)
Defined but never used as a parameter — `LayerKVCache::new` takes individual args. Either integrate it or remove to avoid dead code.

### Verdict: **Approve with minor comments**

---

## PR #31 — `llama-runtime`: MockEngine + CI

**Branch:** `claude/feat-mock-engine` | **Scope:** `crates/llama-runtime/src/lib.rs` (+111), CI config | **Tests:** 4/4 pass

### Summary

`MockEngine` implementing `LlamaEngine` trait via `WhitespaceTokenizer` + `Sampler`. CI updated to trigger on `develop` branch.

### Strengths

- Proves the narrow-waist trait works end-to-end without a real model
- `Send + Sync` verified by test
- Clean dependency layering: runtime → engine + tokenizer + sampling
- Minimal, correct CI change

### Issues

#### 1. `decode()` discards sampled token (minor)
`lib.rs:66-73` — `_token` is computed then thrown away. `TokenStream` is a unit struct with no data. Mock decode is effectively a no-op. Consider storing generated tokens or documenting the limitation.

#### 2. `prefill()` is a complete no-op (acceptable)
Returns `Ok(PrefillResult)` without recording anything. Combined with decode issue, mock can't demonstrate a generate loop. Fine as trait-satisfaction proof.

#### 3. `Sampler` not configurable (nit)
Default settings only (temp=1.0, seed=42). No constructor accepting a `Sampler`. Consider exposing this for test flexibility.

#### 4. `embed()` returns identical zero vectors (nit)
`vec![0.0; 128]` for every input. Hash-based distinct vectors would be more useful for downstream testing.

### Verdict: **Approve**

---

## Cross-PR Observations

1. **Dependency chain is clean** — #31 correctly merges #29 and #30
2. **Merge order matters** — both #29 and #30 carry tokenizer rewrite. Merge #29 first, rebase #30 onto updated main to avoid conflicts
3. **KV cache unused by MockEngine** — #31 depends on #30 but doesn't import `llama-kv`. KV cache integration remains untested at system level
4. **No `unsafe`, no `unwrap()` in library code** (except `Mutex::lock().unwrap()` in mock — acceptable)
5. **42 total tests** across workspace — solid foundational coverage
