# Milestone A: "Hello Inference" (CPU only)

**Goal:** Deterministically generate tokens from a tiny model.

## Status overview

| Milestone | Status | Notes |
|-----------|--------|-------|
| A — "Hello Inference" | In Progress | Foundation complete, CLI demo remaining |
| B — KV Cache Correctness | Not started | KV cache infra landed (PR #18); correctness validation not started |
| C — Real Weight Loading | Not started | |
| D — Metal Enablement | Not started | |
| E — RAG + Agents | Not started | |

See **`docs/PROGRESS_REPORT.md`** for full checklist with PR numbers and Issue #2 epic status.

### Next: CLI `llama-cli generate`

Wire tokenizer → model → sampler → detokenizer into a CLI command. Model blocks (PR #37 merged) and KV equivalence verifier (PR #40 merged) provide the building blocks.

## Checklist

- [x] `llama-engine` trait + session struct — PR #16 (merged), #19 (open)
- [x] Workspace scaffolding — PR #16 (merged)
- [x] Tokenizer trait + reference impl — PR #17, #22 (merged)
- [x] Sampling crate — PR #29 (merged)
- [x] KV cache — PR #18 (merged), PR #46 (atomic session ops, open)
- [x] Model blocks (LLAMA-005) — PR #37 (merged)
- [x] KV equivalence verifier (LLAMA-006) — PR #40 (merged)
- [x] Phase 1 golden tests — PR #47 (open)
- [ ] **CLI `llama-cli generate`** — not started

## Tests

- **Unit:** tokenizer roundtrip (ASCII, unicode, emoji, CJK)
- **Model blocks:** RMSNorm, RoPE, attention, MLP reference golden tests
- **KV equivalence (LLAMA-006):** full forward vs prefill+decode ≤ 1e-5 diff
- **KV off-by-one detection:** injected bug produces > 1e-4 diff
- **Sampling:** determinism, greedy, top-k/p, edge cases
- **Golden (pending):** end-to-end tiny model forward pass vs Python reference

## Key Design Decisions

- **Model format:** safetensors first; gguf later if needed
- **Quantization:** fp16 first; add Q4/Q8 with separate kernel paths later
- **Attention:** non-fused first; APIs designed so flash/fused kernels can replace later
- **Session API:** streaming + cancellation from day 1
- **Observability:** TTFT, tok/s, peak memory, KV bytes as structured events
