# Test Strategy (TDD)

Status: Draft (created 2026-02-15)

Goal: enable fast feedback while building an FFI-backed inference runtime.

## Testing Pyramid

1. Unit tests (fast, deterministic)
- Validate error mapping, config validation, cancellation/backpressure behavior
- Use fake backends for runtime tests where possible

2. Property tests (robustness)
- Tokenization round-trips and invariants (when tokenizer is available)
- Fuzz unsafe boundary inputs (null pointers, invalid lengths) via safe wrappers, not raw `sys`

3. Integration tests (real llama.cpp)
- Gate behind a feature, e.g. `--features llama-cpp`
- Run end-to-end: load model, ingest prompt, generate N tokens, assert basic invariants

4. Performance tests (regression gates)
- Criterion benchmarks for token throughput/latency
- Optional flamegraph capture (not required for CI)

## Fixtures

To keep CI and local dev reliable:
- Prefer a small GGUF fixture (tiny model) stored outside git by default and downloaded in CI, or checked in only if size is acceptable.
- For tests that must be hermetic, use "mock backend" tests that don't require llama.cpp.

## TDD Workflow For New Capability

For each feature slice:
1. Write a failing unit test for the new API contract (validation, error types, or state transition).
2. If FFI is involved, add an integration test that exercises the happy path and one failure path.
3. Add a micro-benchmark only after correctness is established.

## What To Test First (Order)

1. Typed errors replace `String` in `Model::load` and `Session::new`.
2. Resource lifecycle: model/context/session drop semantics.
3. Determinism knobs: seed + fixed sampling produces stable token IDs for a known fixture (when feasible).
4. Cancellation: generation stops quickly and does not corrupt state.
5. Backpressure: streaming consumer slowdown does not grow memory without bound.

## CI Expectations (Proposed)

- Default: `cargo test` runs unit tests only (no llama.cpp dependency).
- Optional job: `cargo test --features llama-cpp` runs integration tests on supported runners.
- Perf: run on demand (manual or nightly), not per PR initially.

