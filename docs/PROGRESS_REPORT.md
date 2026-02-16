# Progress Report: Issues #1 and #2

*Tracks architecture (Issue #1) and epic/user-story plan (Issue #2) for stevedores-org/llama.rs.*

---

## Issue #1 — Architecture + TDD Plan

**Status:** Milestone A in progress, foundation complete.

### Milestones overview

| Milestone | Status | Notes |
|-----------|--------|-------|
| A — "Hello Inference" | In Progress | Foundation complete, CLI demo remaining |
| B — KV Cache Correctness | Not started | KV cache infra landed (PR #18); correctness validation not started |
| C — Real Weight Loading | Not started | |
| D — Metal Enablement | Not started | |
| E — RAG + Agents | Not started | |

### Milestone A checklist

| Item | Status |
|------|--------|
| llama-engine trait + session struct | PR #16 (merged), PR #19 (open) |
| Workspace scaffolding | PR #16 (merged) |
| Tokenizer trait + reference impl | PR #17 (merged), PR #22 (merged) |
| Sampling crate | PR #29 (merged) |
| KV cache | PR #18 (merged), PR #46 (atomic ops, open) |
| Model blocks (LLAMA-005) | PR #37 (merged) |
| KV equivalence verifier (LLAMA-006) | PR #40 (merged) |
| Phase 1 golden tests | PR #47 (open) |
| **CLI `llama-cli generate`** | **Not started** |
| Architecture + TDD docs | PR #3 (merged) |

---

## Issue #2 — Epic and User Story Plan (LLAMA-001 through LLAMA-012)

All 12 issues created and tracked.

### Epic 1: Repository Foundation (3 stories)

| Story | Issue | Status |
|-------|-------|--------|
| LLAMA-001 | #4 | Partially done — workspace + trait landed (PR #16), session fixes in PR #19 |
| LLAMA-002 | #5 | Partially done — trait + reference impl merged (PR #17/#22), streaming UTF-8 not yet |
| LLAMA-003 | #6 | Done — PR #29 merged (greedy, top-k/p, temperature, repetition penalty) |

### Epic 2: Inference & KV Cache (3 stories)

| Story | Issue | Status |
|-------|-------|--------|
| LLAMA-004 | #7 | PR #18 merged, PR #46 open (atomic session ops) |
| LLAMA-005 | #8 | Done — PR #37 merged (RMSNorm, RoPE, SwiGLU, attention, safetensors) |
| LLAMA-006 | #9 | Done — PR #40 merged (KV equivalence true-test) |

### Epic 3–5 (6 stories)

All open, not started.

---

## PRs summary

| State | Count | PRs |
|-------|-------|-----|
| Merged | 13 | #3, #16, #17, #18, #22, #23, #24, #25, #26, #29, #33, #34, #35, #37, #40 |
| Open | 7 | #19, #30, #31, #32, #41, #46, #47 |
| Closed | 2 | #27, #38 |

---

## Key observations

1. **Foundation complete** — workspace, docs, tokenizer, sampling, KV cache, model blocks all landed.
2. **KV equivalence verified** — PR #40 proves prefill+decode matches full forward (LLAMA-006 gate).
3. **Model blocks done** — PR #37 implements RMSNorm, RoPE, SwiGLU, attention with reference golden tests.
4. **Remaining for Milestone A** — CLI demo (`llama-cli generate`), end-to-end integration test.
5. **PR triage needed** — several overlapping open PRs (#30 vs #18, #31, #32). Consolidate or close stale ones.
