# Roadmap (Epic + User Stories)

Status: Open (created 2026-02-15)

This roadmap aligns with the architectural directive: build a **modular, resilient, and accelerated inference runtime** in Rust, and port product behavior from **llama.app** into **stevedores-org/llama.rs**.

Tracking:
- GitHub Issue #1: community-stevedores-org (repo bootstrap / meta)
- GitHub Issue #2: Feature: Architecture + TDD plan to port llama.app -> Rust

## Epic 0: Repo Baseline (Now)

Goal: establish a delivery loop with crisp module boundaries and tests as the default.

User stories:
1. As a contributor, I can run `cargo test` and get deterministic results on a clean machine.
   - Acceptance: `cargo test` passes; docs describe required env vars/features; no network required for unit tests.
2. As a maintainer, I can see the intended architecture and test strategy in-repo.
   - Acceptance: `docs/ARCHITECTURE.md` and `docs/TEST_STRATEGY.md` exist and are referenced by `README.md`.

## Epic 1: llama.cpp Integration (FFI + Safe Wrappers)

Goal: ship a safe, ergonomic Rust API over llama.cpp with a strict separation between unsafe and safe code.

User stories:
1. As a user, I can load a GGUF model and run a single prompt to completion.
   - Acceptance: `Model::load(...)` works; `Session` can generate tokens; errors are typed (not `String`).
2. As a user, I can stream tokens and cancel generation.
   - Acceptance: streaming API supports backpressure; cancellation stops promptly and frees resources.
3. As a user, I can control sampling and context.
   - Acceptance: top-k/top-p/temperature/repetition penalties; context length; seed; n_threads; batch size.

## Epic 2: Resilient Runtime (Scheduling, Backpressure, Observability)

Goal: a runtime that behaves predictably under load and failure.

User stories:
1. As an operator, I can run concurrent requests without memory blowups.
   - Acceptance: bounded queues; request limits; graceful overload behavior.
2. As a developer, I can understand and debug performance regressions.
   - Acceptance: structured logs + spans; metrics for latency/token/s throughput; optional flamegraph hooks.

## Epic 3: Accelerated Inference (CPU/GPU Offload, Batching, KV Cache)

Goal: enable fast inference on common targets while keeping the API stable.

User stories:
1. As a user, I can choose compute backend (CPU/Metal/CUDA where supported by llama.cpp build).
   - Acceptance: feature flags for backends; runtime errors are explicit; graceful fallback if unsupported.
2. As a user, I get better throughput via batching.
   - Acceptance: batched decode path; benchmarks show throughput improvement on representative workloads.
3. As a user, I can reuse KV cache and do multi-turn chat efficiently.
   - Acceptance: session supports prompt caching; clear invalidation semantics; memory is bounded/configurable.

## Epic 4: Port llama.app UX (CLI First, then Server)

Goal: reproduce llama.app behavior as a Rust-delivered product surface without forcing a particular UI.

User stories:
1. As a user, I can use a CLI that matches llama.app behavior for common flows.
   - Acceptance: `llama-cli` supports model selection, prompt/chat modes, streaming, and config file.
2. As an integrator, I can run a local HTTP server for programmatic use.
   - Acceptance: minimal REST API; streaming responses; request cancellation; compatibility notes documented.

## Milestones (Proposed)

M0 (Docs + skeleton):
- Planning docs landed; crate layout decision; error types; test scaffolding present.

M1 (Single prompt end-to-end):
- FFI minimal surface + safe wrapper; load model; generate tokens; integration test with a tiny fixture.

M2 (Streaming + cancellation + config):
- streaming API; cancellation; CLI config; basic sampling controls; structured logs.

M3 (Runtime hardening):
- concurrency limits; backpressure; metrics; soak tests; perf baseline.

M4 (Acceleration + batching):
- backend selection; batched decode; KV cache reuse; benchmarks + regression gates.

