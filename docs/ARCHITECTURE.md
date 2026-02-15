# Architecture

Status: Draft (created 2026-02-15)

Directive: build a **modular, resilient, and accelerated inference runtime** in Rust, with a path to port **llama.app** behavior.

## Design Principles

- Unsafe code is quarantined to a small surface and wrapped in typed, safe APIs.
- API stability: keep public interfaces narrow, versioned, and testable.
- Resilience by default: cancellation, timeouts, backpressure, bounded memory.
- Performance is a feature: avoid unnecessary copies, allow zero/low-copy IO (mmap), and expose tuning knobs.

## Proposed Crate Layout (Workspace-ready)

This repo currently builds as a single crate. As soon as FFI starts, convert to a workspace:

- `crates/llama-sys`
  - Raw `extern "C"` bindings to llama.cpp
  - Build integration (cmake/cc) and feature-gated backend compilation
  - No allocations beyond what's required; minimal safety assumptions

- `crates/llama-core`
  - Safe wrappers: `Model`, `Context`, `Session`
  - Typed errors, RAII, thread-safety story documented per type
  - Tokenization, sampling config, logits access, KV cache controls

- `crates/llama-runtime`
  - Request scheduler, batching, cancellation, resource governance
  - Backpressure primitives and memory accounting
  - Observability hooks (tracing + metrics)

- `crates/llama-cli` (or keep `src/bin/cli.rs` initially)
  - Human-facing interface; configuration file; common flows

- `crates/llama-server` (later)
  - HTTP API, streaming responses, cancellation, auth (if needed)

## Key Abstractions (Targets)

### `Model`

Responsibilities:
- Own the underlying llama model handle
- Manage model-level options (mmap, vocab access, metadata)

Invariants:
- `Model` is an owning handle; drop frees model resources exactly once.

### `Session`

Responsibilities:
- Own a context and KV cache state for multi-turn interactions
- Provide generation methods: `generate`, `stream`, `cancel`
- Expose prompt ingestion and incremental decode

Invariants:
- Cancellation is cooperative and prompt; a cancelled session can either be reused (clear semantics) or must be recreated.
- Bounded memory: context size and KV cache behavior are explicit and configurable.

### `Runtime`

Responsibilities:
- Accept requests, schedule them, and enforce limits
- Optional batching (same model) and prioritized queues
- Provide observability: per-request spans, token throughput, queue depth

Invariants:
- No unbounded queues.
- Every request has: deadline, cancellation token, and a defined overload behavior.

## Acceleration Strategy (Surface, not policy)

The Rust API should expose capability discovery rather than hard-coding platform rules:
- Backend/features: CPU, Metal, CUDA (depending on llama.cpp build)
- Threads/batch sizing knobs
- Optional GPU layer offload controls (as supported by llama.cpp)

## Compatibility With llama.app (Porting Plan)

Treat llama.app as a behavioral spec:
- Identify user-visible flows (chat, single prompt, streaming, stop sequences, system prompts, config)
- Reproduce behavior in `llama-cli` first
- Only then lock a server API

Deliverables for a first parity pass:
- A config schema that can represent llama.app defaults
- A CLI UX that can run in both interactive and non-interactive modes

