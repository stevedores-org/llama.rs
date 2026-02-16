# Code Review: PR #3 — `docs: roadmap + architecture + TDD plan (issues #1/#2)`

**Reviewer**: Claude (automated)
**PR**: stevedores-org/llama.rs#3
**Branch**: `chore/planning-docs` → `develop`
**Author**: stevei101

---

## Summary

This PR adds three planning documents (`docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/TEST_STRATEGY.md`) and links them from `README.md`. It addresses issues #1 (repo bootstrap) and #2 (architecture + TDD plan). No code changes—purely documentation.

**Verdict: Approve with requested changes.** The documents are a solid foundation, but there are meaningful gaps between what these docs describe and what the open issues (especially #4 through #15) actually require. Addressing the items below before merge will prevent the docs from going stale on day one.

---

## What's Good

1. **Clear structure.** Three focused docs with distinct scopes (roadmap, architecture, test strategy) is the right decomposition. Linking them from `README.md` makes discovery easy.

2. **Design principles are sound.** Quarantined unsafe code, narrow public APIs, bounded memory, and resilience-by-default are all appropriate for an FFI-backed inference runtime.

3. **Test pyramid is pragmatic.** The unit → property → integration → perf progression with feature-gated llama.cpp tests is a sensible approach for a project that will have heavy native dependencies.

4. **Milestones are sequenced correctly.** M0 → M4 follows a natural progression from skeleton through single-prompt, streaming, runtime hardening, and acceleration.

---

## Issues to Address

### 1. Architecture diverges from the issue tracker (High)

**`ARCHITECTURE.md` proposes 5 crates:**
`llama-sys`, `llama-core`, `llama-runtime`, `llama-cli`, `llama-server`

**Issue #1 and #4 specify 8+ crates with different names:**
`llama-engine`, `llama-models`, `llama-runtime`, `llama-kv`, `llama-tokenizer`, `llama-sampling`, `llama-server`, `llama-rag`, `llama-agents`

These are fundamentally different module decompositions. `ARCHITECTURE.md` merges tokenization, sampling, KV cache, and model definitions into `llama-core`, while the issues break them into separate crates. The "narrow waist" `LlamaEngine` trait from issue #4 is absent from the architecture doc entirely.

**Requested change:** Either align the architecture doc with the issue tracker's crate layout, or explicitly document why the doc departs from it (e.g., "we'll start with a coarser decomposition and split later"). The current silent inconsistency will confuse contributors.

### 2. Missing `LlamaEngine` trait — the "narrow waist" (High)

Issue #4 (`[LLAMA-001]`) defines the core deliverable as a `LlamaEngine` trait with `load_model`, `prefill`, `decode`, and `embed` methods. This is the central abstraction that `oxidizedMLX` and `oxidizedRAG` depend on (per issue #4's user story). `ARCHITECTURE.md` describes `Model`, `Session`, and `Runtime` but never mentions `LlamaEngine` or the trait-based dispatch pattern.

**Requested change:** Add the `LlamaEngine` trait to the "Key Abstractions" section, or explain its relationship to the `Model`/`Session`/`Runtime` types. This is the most important interface in the system per the issue tracker.

### 3. Roadmap epics don't align with issue tracker epics (Medium)

The PR's `ROADMAP.md` defines Epics 0–4 with a generic FFI/runtime/acceleration/CLI framing. The issue tracker defines Epics 1–5 with specific scope:

| ROADMAP.md | Issue tracker |
|---|---|
| Epic 0: Repo baseline | (no issue equivalent) |
| Epic 1: llama.cpp Integration | Epic 1: Foundation & Core Engine (#4, #5, #6) |
| Epic 2: Resilient Runtime | Epic 2: Inference Logic & KV Cache (#7, #8, #9) |
| Epic 3: Accelerated Inference | Epic 3: Hardware Acceleration (#10, #11) |
| Epic 4: Port llama.app UX | Epic 4: Application Surface (#12, #13) |
| (none) | Epic 5: Agentic Compute (#14, #15) |

Epic 5 (agentic compute + data fabric, issues #14/#15) is entirely absent from `ROADMAP.md`. The epic numbering is also off-by-one since the roadmap starts at 0 while issues start at 1.

**Requested change:** Reconcile epic numbering and scope with the issue tracker. Either adopt the same numbering (Epics 1–5) or add a mapping table. Add Epic 5 content or note it as out-of-scope for now.

### 4. oxidizedMLX / oxidizedRAG / oxidizedgraph dependencies not mentioned (Medium)

Issue #1 explicitly names `oxidizedMLX`, `oxidizedRAG`, and `oxidizedgraph` as core dependencies and integration targets. The architecture and roadmap docs don't mention any of them. These external crate dependencies shape the trait boundaries, build system, and acceleration strategy significantly.

**Requested change:** Add a "Dependencies / Integration Points" section to `ARCHITECTURE.md` covering the oxidized ecosystem crates, or explain that the docs describe a standalone-first approach.

### 5. `TEST_STRATEGY.md` doesn't cover all testing tiers from issue #1 (Medium)

Issue #1 specifies five testing tiers: unit, property, **golden tests (Python reference comparison)**, **backend parity tests**, and performance benchmarks. The PR's `TEST_STRATEGY.md` covers four tiers (unit, property, integration, performance) but omits:

- **Golden tests** — comparing Rust output against a Python reference implementation for bit-exactness
- **Backend parity tests** — verifying CPU vs. Metal vs. CUDA produce equivalent results

These are listed as "True Tests" / critical correctness gates in issue #2.

**Requested change:** Add golden tests and backend parity tests to the testing pyramid, even if just as future tiers with placeholder descriptions.

### 6. `Session::Default` impl will panic (Low — existing code, not this PR)

Not introduced by this PR, but worth noting: `src/session.rs:17` implements `Default` by calling `.expect()` on `Session::new()`, which currently always returns `Err`. This means `Session::default()` panics unconditionally. The test strategy doc says `cargo test` should pass on a clean machine, but if anyone writes a test using `Session::default()`, it will panic.

**Suggested follow-up:** Either remove the `Default` impl or gate it. This aligns with the test strategy doc's "typed errors replace `String`" first priority.

### 7. Follow-up items in PR description should be tracked (Low)

The PR description mentions two follow-up items:
1. Convert to workspace when FFI begins
2. Replace `String`-based errors with typed errors

These align with issues #4 and the test strategy but aren't linked to specific issues.

**Suggested:** Add these as items in the relevant issues (#4 for workspace, test strategy for typed errors) so they don't get lost.

---

## Nits

- `ARCHITECTURE.md` line 1: "Status: Draft (created 2026-02-15)" — consider using a format that auto-updates or is easily grep-able (e.g., a front-matter block).
- `ROADMAP.md` tracking section references "GitHub Issue #1" and "#2" by description rather than linking them. Use `[#1](https://github.com/stevedores-org/llama.rs/issues/1)` for clickable references.
- `TEST_STRATEGY.md` "What To Test First" section overlaps with `ROADMAP.md` M0/M1 scope but uses different ordering. Consider cross-referencing milestones explicitly.
- `README.md` diff adds the planning docs section between "Overview" and "Getting Started" — this is fine, but consider placing it after "Features" so the quick-start path isn't interrupted for new visitors.

---

## Files Reviewed

| File | Status | Notes |
|---|---|---|
| `README.md` | Modified | Adds planning docs links — OK |
| `docs/ARCHITECTURE.md` | New | Needs alignment with issue tracker (items 1, 2, 4) |
| `docs/ROADMAP.md` | New | Needs alignment with issue tracker (items 3, 4) |
| `docs/TEST_STRATEGY.md` | New | Missing golden + parity tiers (item 5) |

---

## Recommendation

**Approve with changes.** The documentation quality is good and the direction is right. The primary concern is that these planning docs diverge from the issue tracker on crate layout, trait design, epic scope, and testing tiers. Merging as-is means the docs are partially outdated before any code lands. Addressing items 1–5 above would make this a strong foundation that contributors and the oxidized ecosystem integrations can rely on.
