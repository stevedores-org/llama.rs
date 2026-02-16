# Test Strategy (TDD)

Status: Draft (updated 2026-02-15)

Goal: enable fast feedback while building a modular inference runtime with oxidizedMLX backends.

## Testing Pyramid

### 1. Unit tests (fast, deterministic) — always on PR
- Validate error mapping, config validation, cancellation/backpressure behavior
- Tokenization roundtrip and edge cases
- Sampling determinism (seeded RNG)
- Use fake backends for runtime tests where possible

### 2. Property tests (robustness) — on PR
- Tokenization invariants: shapes, encoding/decoding edge cases
- KV cache append-only invariant: cache length increases by 1 per decode
- Graph arena: node count doesn't grow unbounded per token

### 3. Golden tests (Python reference comparison) — PR small, nightly full
- Tiny forward pass matches Python reference outputs within tolerance (`1e-5`)
- Logits for a fixed prompt match reference
- Tokenization outputs match HF tokenizer for a fixed set of prompts
- End-to-end "1-turn chat": fixed prompt + weights → first N tokens match reference

### 4. Backend parity tests (Metal vs CPU) — nightly on Apple Silicon
- Op-level: Metal matmul/attention result == CPU result (within float tolerance)
- Attention block "Golden Test" passes across all enabled backends
- Stress: concurrent sessions on same device don't deadlock

### 5. Performance tests (regression gates) — nightly/weekly
- Criterion benchmarks for token throughput/latency
- TTFT (time to first token) tracking
- Memory footprint regression detection
- Optional flamegraph capture (not required for CI)

## True Tests (Critical Correctness Gates)

These are enforced in CI as hard gates:

1. **Bit-Perfect Tokenization:** `detokenize(tokenize(x)) == x` (allowing normalization)
2. **KV Equivalence:** `full_forward(prompt) logits == prefill(prompt[:-1]) + decode(last_token) logits`
3. **Cross-Backend Parity:** Metal result == CPU result (within float tolerance)
4. **Cancellation Safety:** Dropping a `Future`/`TokenStream` terminates graph execution immediately

## Fixtures

- Prefer small safetensors fixtures (tiny model) for golden tests
- For hermetic tests, use mock/stub backends (no model files required)
- `tests/fixtures/` directory for tiny models, tokenizers, safetensors fixtures
- `tools/py_ref/` for Python reference test generation scripts

## TDD Workflow For New Capability

For each feature slice:
1. Write a failing unit test for the new API contract (validation, error types, or state transition).
2. If a golden reference exists, add a golden test comparing against Python outputs.
3. If backend-specific, add a parity test (Metal vs CPU).
4. Add a micro-benchmark only after correctness is established.

## What To Test First (Order)

1. Tokenizer roundtrip: `detokenize(tokenize(x)) == x`
2. Sampling determinism: fixed logits + seed → fixed sequence
3. KV cache equivalence: prefill + 1-step decode == full forward
4. Attention block golden: fixed Q/K/V → compare SDPA output vs numpy reference
5. Cancellation: generation stops quickly and does not corrupt state

## CI Expectations

- **Per PR:** `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test --workspace`
- **Nightly:** golden tests (full suite), backend parity (Apple Silicon), performance benchmarks
- **On demand:** soak tests, memory profiling
