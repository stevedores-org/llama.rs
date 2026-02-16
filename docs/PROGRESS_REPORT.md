# Progress Report: Issues #1 and #2

*Tracks architecture (Issue #1) and epic/user-story plan (Issue #2) for stevedores-org/llama.rs.*

---

## Issue #1 — Architecture + TDD Plan

**Status:** Milestone A in progress, foundational work landed.

### Milestones overview

| Milestone | Status | Notes |
|-----------|--------|-------|
| A — "Hello Inference" | In Progress | Scaffold done, tokenizer done, remaining items open |
| B — KV Cache Correctness | Not started | PR #18 open but unmerged |
| C — Real Weight Loading | Not started | |
| D — Metal Enablement | Not started | |
| E — RAG + Agents | Not started | |

### Milestone A checklist

| Item | Status |
|------|--------|
| llama-engine trait + session struct | PR #16 (merged), PR #19 (review fixes, open) |
| Workspace scaffolding | PR #16 (merged) |
| Tokenizer trait + reference impl | PR #17 (merged), PR #22 (review fix, merged) |
| Sampling crate | PR #21 (open) |
| KV cache | PR #18 (open) |
| **Tiny model forward pass** | **Not started** |
| CLI `llama-cli generate` | Not started |
| Architecture + TDD docs | PR #3 (merged) |

---

## Issue #2 — Epic and User Story Plan (LLAMA-001 through LLAMA-012)

All 12 issues created and tracked.

### Epic 1: Repository Foundation (3 stories)

| Story | Issue | Status |
|-------|-------|--------|
| LLAMA-001 | #4 | Partially done — workspace + trait landed (PR #16), session fixes in PR #19 |
| LLAMA-002 | #5 | Partially done — trait + reference impl merged (PR #17/#22), streaming UTF-8 not yet |
| LLAMA-003 | #6 | In progress — PR #21 open |

### Epic 2: Inference & KV Cache (3 stories)

| Story | Issue | Status |
|-------|-------|--------|
| LLAMA-004 | #7 | PR #18 open (needs review) |
| LLAMA-005 | #8 | Not started |
| LLAMA-006 | #9 | Not started |

### Epic 3–5 (6 stories)

All open, not started.

---

## PRs summary

| State | Count | PRs |
|-------|-------|-----|
| Merged | 5 | #3, #16, #17, #22, #24 |
| Open | 7 | #18, #19, #21, #23, #25, #26, #27 |

---

## Key observations

1. **Good velocity on foundation** — workspace, docs, and tokenizer all landed in day 1.
2. **PR pile-up** — 7 open PRs; some overlapping (e.g. #25, #26, #27 all touch tokenizer/engine). Triage and merge or close to avoid conflict churn.
3. **Multiple agents contributing** — PRs from `feat/`, `codex/`, `cursor/`, and standalone branches. Coordination would help avoid duplicate work.
4. **Milestone A ~40% complete** — scaffold + tokenizer done; sampling/KV/model/CLI remaining.
