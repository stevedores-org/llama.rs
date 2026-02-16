# Roadmap (Epic + User Stories)

Status: Open (updated 2026-02-15)

This roadmap aligns with the architectural directive: build a **modular, resilient, and accelerated inference runtime** in Rust using oxidizedMLX, oxidizedRAG, and oxidizedgraph.

Tracking:
- **Primary planning tracker:** [Issue #2](https://github.com/stevedores-org/llama.rs/issues/2) (Epic and User Story plan)
- **Supporting architecture context:** [Issue #1](https://github.com/stevedores-org/llama.rs/issues/1)

---

## Epic 1: Repository Foundation & Core Engine Abstraction

Goal: establish the Rust workspace, define the core traits that mediate between model logic and hardware, and implement deterministic tokenization.

| Story | Issue | Description |
|-------|-------|-------------|
| LLAMA-001 | [#4](https://github.com/stevedores-org/llama.rs/issues/4) | Modular workspace and narrow-waist `LlamaEngine` trait |
| LLAMA-002 | [#5](https://github.com/stevedores-org/llama.rs/issues/5) | Robust tokenizer crate with streaming UTF-8 support |
| LLAMA-003 | [#6](https://github.com/stevedores-org/llama.rs/issues/6) | Foundational sampling crate (greedy, top-k/p, temperature) |

## Epic 2: Inference Logic & KV Cache Management

Goal: implement graph-friendly model architectures and first-class memory management for token sequences.

| Story | Issue | Description |
|-------|-------|-------------|
| LLAMA-004 | [#7](https://github.com/stevedores-org/llama.rs/issues/7) | First-class KV Cache (prefill, decode, paging) |
| LLAMA-005 | [#8](https://github.com/stevedores-org/llama.rs/issues/8) | Llama 3 and Qwen model blocks in Rust |
| LLAMA-006 | [#9](https://github.com/stevedores-org/llama.rs/issues/9) | Prefill vs. Decode verification suite (KV equivalence) |

## Epic 3: Hardware Acceleration & Backend Parity

Goal: integrate oxidizedMLX and enforce strict parity between CPU and Metal backends.

| Story | Issue | Description |
|-------|-------|-------------|
| LLAMA-007 | [#10](https://github.com/stevedores-org/llama.rs/issues/10) | Runtime selector for oxidizedMLX backends |
| LLAMA-008 | [#11](https://github.com/stevedores-org/llama.rs/issues/11) | Backend Parity Gate (Metal vs CPU golden tests) |

## Epic 4: Application Surface & OpenAI-Compatible Server

Goal: expose the engine via a high-performance HTTP server with streaming, sessions, and cancellation.

| Story | Issue | Description |
|-------|-------|-------------|
| LLAMA-009 | [#12](https://github.com/stevedores-org/llama.rs/issues/12) | OpenAI-compatible HTTP server (`/v1/chat/completions`) |
| LLAMA-010 | [#13](https://github.com/stevedores-org/llama.rs/issues/13) | Session and cancellation management |

## Epic 5: Agentic Compute & Data Fabric Integration

Goal: integrate with the broader "oxidized" ecosystem for RAG and multi-agent orchestration.

| Story | Issue | Description |
|-------|-------|-------------|
| LLAMA-011 | [#14](https://github.com/stevedores-org/llama.rs/issues/14) | RAG adapter for oxidizedRAG |
| LLAMA-012 | [#15](https://github.com/stevedores-org/llama.rs/issues/15) | Agent orchestration via oxidizedgraph |

---

## Milestones

**Milestone A — "Hello Inference" (CPU only)**
- Workspace scaffold + `LlamaEngine` trait *(done: [PR #16](https://github.com/stevedores-org/llama.rs/pull/16))*
- Tokenizer loads + roundtrip tests
- Tiny model forward (single block) on CPU
- Greedy sampling
- CLI: `llama-cli generate --prompt "hi"`

**Milestone B — KV Cache Correctness (prefill + decode)**
- KV cache struct + layout
- Prefill writes cache; decode uses cache
- Streaming `TokenStream`
- KV equivalence test: `full_forward == prefill + decode`

**Milestone C — Real Weight Loading (safetensors)**
- oxidizedMLX `mlx-io` safetensors integration
- Tensor name mapping + shape/dtype validation
- Memory footprint sanity checks

**Milestone D — Metal Enablement Behind Parity Gate**
- Required op set for chosen model family
- Parity suite (Metal vs CPU) for ops and attention block
- Feature flag gating default backend

**Milestone E — RAG + Agents (Product)**
- oxidizedRAG adapter with citation prompt builder
- oxidizedgraph agent nodes: Retrieve → Rerank → Prompt → Generate
