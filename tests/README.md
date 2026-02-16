# Phase 1 Test Suite for llama.rs

This directory contains the Phase 1 test suite that enforces the "True Tests" and acceptance criteria from Issue #2.

## Architecture

The test suite is organized by concern:

### 1. Tokenizer Tests (`tokenizer/`)
- **roundtrip**: `detokenize(tokenize(x)) == x` (True Test #1)
- **streaming**: UTF-8 boundary handling for multi-byte characters
- **property tests**: Random unicode strings and edge cases
- **asset loading**: Validates tokenizer.json loading

### 2. Sampler Tests (`sampler/`)
- **determinism**: Seeded RNG produces identical results
- **correctness**: Greedy, Top-K, Top-P, Temperature implementations
- **edge cases**: top_k=1, top_p=0.0, temperature=0
- **distributions**: Entropy scaling with temperature

### 3. KV Cache & Inference Tests (`inference/`)
- **kv_equivalence**: Full forward vs prefill+decode (True Test #2)
- **rope_offset**: Position encoding correctness in decode phase
- **shape_invariants**: Pre-alloc dimensions and write pointer correctness

### 4. Model Block Golden Tests (`blocks/`)
- **rmsnorm**: Golden vector comparison against reference
- **rope**: Small QK vectors with known sin/cos tables
- **mlp**: Two-layer MLP with known weights
- **attention**: Tiny attention case with reference implementation

### 5. Backend Parity Tests (`backend_parity/`)
- **cpu_vs_metal**: Output comparison within tolerance
- **kernel_matrix**: Runtime validation of required ops
- **concurrent_sessions**: Stress test for deadlock freedom

### 6. Cancellation Safety Tests (`cancellation/`)
- **cancel_flag**: AtomicBool stops generation promptly
- **drop_stream**: Dropping receiver terminates inference thread
- **kv_cleanup**: Memory released after cancellation

## Test Fixtures

### Toy Model (`fixtures/toy_model.safetensors`)
- Small dimensions (d_model=64, n_heads=2)
- Fixed seed for reproducibility
- Includes reference logits for verification

### Golden Cases (`fixtures/tokenizer_cases.jsonl`)
- ASCII, whitespace, unicode edge cases
- Chat-template formatted strings
- Streaming boundary cases

## Running Tests

```bash
# Run all tests
cargo test --workspace

# Run specific suite
cargo test --test tokenizer_
cargo test --test inference_
cargo test --test backend_parity_ --features metal

# Run with logging
RUST_LOG=debug cargo test --workspace -- --nocapture

# Run deadlock stress test (longer timeout)
cargo test concurrent_sessions -- --test-threads=1 --ignored
```

## CI Runners

- **Linux**: CPU backend, tokenizer, sampler, most unit/integration tests
- **macOS (Apple Silicon)**: Metal backend parity, concurrent stress tests

## Phase 1 Gates

A PR cannot merge if these fail:

1. ✅ `tokenizer_roundtrip_*` - Bit-perfect tokenization
2. ✅ `inference_kv_equivalence` - KV cache equivalence
3. ✅ `backend_parity_*` (macOS) - CPU vs Metal output match
4. ✅ `cancellation_*` - Cancellation safety

## References

- Issue #2: Epic and user story plan with True Tests
- Issue #39: Phase 1 test suite specification
- LLAMA-002/003/004/006/008: Feature requirements with test gates
