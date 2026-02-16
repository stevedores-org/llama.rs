# Milestone A: "Hello Inference" (CPU only)

**Goal:** Deterministically generate tokens from a tiny model.

## Checklist

- [x] `llama-engine` trait + session struct
- [x] Workspace scaffolding with all crates
- [ ] Tokenizer loads + tokenize/detokenize roundtrip tests
- [ ] Tiny model forward (single block) on CPU backend
- [ ] Greedy sampling works
- [ ] CLI can run: `llama-cli generate --prompt "hi"`

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
