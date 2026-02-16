# Milestone A: "Hello Inference" (CPU only)

**Goal:** Deterministically generate tokens from a tiny model.

## Status overview

| Milestone | Status | Notes |
|-----------|--------|-------|
| A — "Hello Inference" | In Progress | Scaffold done, tokenizer done, remaining items open |
| B — KV Cache Correctness | Not started | PR #18 merged |
| C — Real Weight Loading | Not started | |
| D — Metal Enablement | Not started | |
| E — RAG + Agents | Not started | |

See **`docs/PROGRESS_REPORT.md`** for full checklist with PR numbers and Issue #2 epic status.

### Next: Tiny model forward pass — not started

Implement a single-block forward pass on the CPU backend so that a tiny model can produce logits for a fixed prompt. This unblocks golden tests (logits vs. Python reference) and `llama-cli generate`.

## Checklist

- [x] `llama-engine` trait + session struct — PR #16 (merged), #19 (open)
- [x] Workspace scaffolding — PR #16 (merged)
- [x] Tokenizer trait + reference impl — PR #17, #22 (merged)
- [x] Sampling crate — PR #21 (open)
- [x] KV cache — PR #18 (merged)
- [ ] **Tiny model forward pass** — single block on CPU backend *(next)*
- [ ] CLI `llama-cli generate` — not started

## Tests

- **Unit:** tokenizer roundtrip
- **Golden:** tiny forward pass matches Python reference
- **Golden:** logits for a fixed prompt match reference (within tolerance)
- **Sampling determinism:** fixed logits + seed → fixed sequence

## Key Design Decisions

- **Model format:** safetensors first; gguf later if needed
- **Quantization:** fp16 first; add Q4/Q8 with separate kernel paths later
- **Attention:** non-fused first; APIs designed so flash/fused kernels can replace later
- **Session API:** streaming + cancellation from day 1
- **Observability:** TTFT, tok/s, peak memory, KV bytes as structured events
