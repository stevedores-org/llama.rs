# Architecture

> Copied from [Issue #1](https://github.com/stevedores-org/llama.rs/issues/1) — the canonical architecture and TDD plan.

## "Narrow Waist" + Replaceable Internals

```
llama.rs/
  crates/
    llama-engine/        # narrow-waist engine trait + streaming API
    llama-models/        # model architectures (Llama/Qwen/Mistral blocks)
    llama-runtime/       # execution: oxidizedMLX integration, backend selection
    llama-tokenizer/     # tokenizers + chat templates
    llama-sampling/      # samplers + penalties + stop conditions
    llama-kv/            # KV cache layouts + paging/eviction
```

## The Narrow Waist: `llama-engine`

Everything else plugs into this.

**Core trait:**
- `load_model(spec) -> ModelHandle`
- `tokenize(text) -> Vec<i32>`
- `detokenize(tokens) -> String`
- `prefill(session, prompt_tokens) -> PrefillResult` — build KV cache
- `decode(session) -> TokenStream` — streaming generation
- `embed(texts) -> Vec<Vec<f32>>` — for oxidizedRAG

**Why this matters:**
- oxidizedRAG and oxidizedgraph both depend on *engine behavior*, not implementation details.
- You can swap CPU/Metal/FFI backends under `llama-runtime` without changing the app.

## Runtime: `llama-runtime`

- Backend selection (`cpu|metal|ffi`) + gating
- Tensor allocator policy
- Kernel availability matrix (what ops are supported on Metal)
- Telemetry hooks (TTFT, tok/s, memory, KV cache size)

## Model Definitions: `llama-models`

Inference-first model blocks:
- Embeddings
- Attention block (with RoPE)
- RMSNorm/LayerNorm
- MLP
- Output head

Graph-friendly (oxidizedMLX lazy graph) but also efficient for decode.

## KV Cache: `llama-kv`

First-class KV cache that supports:
- **Prefill** writes K/V for `[seq, heads, head_dim]`
- **Decode** appends 1 token at a time
- Memory-friendly layout (contiguous by head or by token)
- Future: paging/eviction / sliding window

## Tokenization: `llama-tokenizer`

- Load tokenizer assets (json/sentencepiece/etc.)
- Chat templates per model (system/user/assistant format)
- Streaming detokenize (handle partial UTF-8)

## Sampling: `llama-sampling`

- Greedy, temperature, top-k/top-p
- Repetition penalty, stop sequences
- Deterministic seeded RNG for tests

## TDD: True Tests

1. **Bit-Perfect Tokenization:** `detokenize(tokenize(x)) == x`
2. **KV Equivalence:** `prefill + 1-step decode logits == full forward logits`
3. **Cross-Backend Parity:** Metal result == CPU result (within float tolerance)
4. **Cancellation Safety:** Dropping a Future/TokenStream terminates graph execution immediately
