# Architecture

Status: Draft (updated 2026-02-15)

Directive: build a **modular, resilient, and accelerated inference runtime** in Rust, using [oxidizedMLX](https://github.com/stevedores-org/oxidizedMLX) for tensor/runtime + Metal, [oxidizedRAG](https://github.com/stevedores-org/oxidizedRAG) for retrieval/memory, and [oxidizedgraph](https://github.com/stevedores-org/oxidizedgraph) for agent orchestration.

The goal is a **real app runtime** (streaming, sessions, KV cache, tools), not just a model runner.

## Design Principles

- **Narrow waist**: the `LlamaEngine` trait is the single stable interface that all consumers depend on.
- Unsafe code is quarantined to a small surface and wrapped in typed, safe APIs.
- API stability: keep public interfaces narrow, versioned, and testable.
- Resilience by default: cancellation, timeouts, backpressure, bounded memory.
- Performance is a feature: avoid unnecessary copies, allow zero/low-copy IO (mmap), and expose tuning knobs.

## Crate Layout

```
llama.rs/
  crates/
    llama-engine/        # narrow-waist engine trait + streaming API
    llama-models/        # model architectures (Llama/Qwen/Mistral blocks)
    llama-runtime/       # execution: oxidizedMLX integration, backend selection
    llama-tokenizer/     # tokenizers + chat templates
    llama-sampling/      # samplers + penalties + stop conditions
    llama-kv/            # KV cache layouts + paging/eviction
  docs/
    ARCHITECTURE.md      # this file
    MILESTONE_A.md       # first milestone checklist
    ROADMAP.md           # epic + user story plan
    TEST_STRATEGY.md     # TDD approach + CI expectations
```

Future crates (not yet scaffolded):
- `llama-server` — OpenAI-compatible HTTP API ([LLAMA-009](https://github.com/stevedores-org/llama.rs/issues/12))
- `llama-cli` — CLI runner for debug + bench
- `llama-rag` — oxidizedRAG adapter ([LLAMA-011](https://github.com/stevedores-org/llama.rs/issues/14))
- `llama-agents` — oxidizedgraph nodes ([LLAMA-012](https://github.com/stevedores-org/llama.rs/issues/15))

## The Narrow Waist: `LlamaEngine` Trait

Everything else plugs into this. Defined in `llama-engine` ([LLAMA-001](https://github.com/stevedores-org/llama.rs/issues/4)):

```rust
pub trait LlamaEngine: Send + Sync {
    fn load_model(&self, spec: &ModelSpec) -> Result<ModelHandle>;
    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;
    fn detokenize(&self, tokens: &[i32]) -> Result<String>;
    fn prefill(&self, session: &mut Session, tokens: &[i32]) -> Result<PrefillResult>;
    fn decode(&self, session: &mut Session) -> Result<TokenStream>;
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}
```

**Why this matters:**
- oxidizedRAG and oxidizedgraph both depend on *engine behavior*, not implementation details.
- You can swap CPU/Metal/FFI backends under `llama-runtime` without changing the app.

## Key Crates

### `llama-engine` — Core types and trait
- `LlamaEngine` trait (the narrow waist)
- `ModelSpec`, `ModelHandle`, `Session`, `PrefillResult`, `TokenStream`
- `LlamaError` typed error enum (not `String`)

### `llama-tokenizer` — Deterministic tokenization ([LLAMA-002](https://github.com/stevedores-org/llama.rs/issues/5))
- Load HF `tokenizer.json` and SentencePiece assets
- Chat templates per model family
- Streaming detokenize (handle partial UTF-8)

### `llama-models` — Model architectures ([LLAMA-005](https://github.com/stevedores-org/llama.rs/issues/8))
- RMSNorm, RoPE, MLP, Attention blocks
- Weight loading from safetensors
- Graph-friendly (oxidizedMLX lazy graph) but efficient for decode

### `llama-runtime` — Backend selection ([LLAMA-007](https://github.com/stevedores-org/llama.rs/issues/10))
- Backend selection (`cpu|metal`) via feature gates
- Tensor allocator policy
- Kernel availability matrix (what ops are supported on Metal)
- Telemetry hooks (TTFT, tok/s, memory, KV cache size)

### `llama-sampling` — Sampling strategies ([LLAMA-003](https://github.com/stevedores-org/llama.rs/issues/6))
- Greedy, temperature, top-k/top-p
- Repetition penalty, stop sequences
- Deterministic seeded RNG for reproducible tests

### `llama-kv` — KV cache management ([LLAMA-004](https://github.com/stevedores-org/llama.rs/issues/7))
- Prefill writes K/V for `[seq, heads, head_dim]`
- Decode appends 1 token at a time
- Memory-friendly layout for Metal/CPU shared memory
- Future: paging/eviction / sliding window

## Acceleration Strategy

The runtime exposes capability discovery rather than hard-coding platform rules:
- Feature gates (`metal`, `cpu`) control backend compilation
- Kernel availability matrix validates op support at startup
- Backend parity gate ([LLAMA-008](https://github.com/stevedores-org/llama.rs/issues/11)): Metal can't be default unless parity tests pass

## Session & Cancellation ([LLAMA-010](https://github.com/stevedores-org/llama.rs/issues/13))

- Cancellation is cooperative and prompt; dropping a `TokenStream` terminates graph execution
- Bounded memory: context size and KV cache behavior are explicit and configurable
- Session ID maps to KV cache lifecycle (freed or archived on completion)
