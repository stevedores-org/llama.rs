# PR #41 Review: merge develop into main

**Author:** stevei101 | **Base:** `main` ← **Head:** `develop`
**Files changed:** 3 (README.md, docs/MILESTONE_A.md, docs/PROGRESS_REPORT.md)

This is a docs-only PR that adds planning/progress tracking documentation and links them from the README. No code changes. Below are the issues found, organized by severity.

---

## Issue 1 (must fix): Sampling crate marked complete but PR is still open

**File:** `docs/MILESTONE_A.md:26`

```markdown
- [x] Sampling crate — PR #21 (open)
```

The checkbox is `[x]` (done) but the annotation says "PR #21 (open)." If the PR hasn't merged, the item isn't complete. This should be `- [ ]` or use a different indicator for in-progress, or at minimum `- [x] ... (merged)` if it has actually merged since this was written.

Note that `docs/PROGRESS_REPORT.md:28` correctly shows this as just `PR #21 (open)` without a done-checkmark, so the two files already disagree on this item's status.

**Suggested fix in `docs/MILESTONE_A.md`:**
```markdown
- [ ] Sampling crate — PR #21 (open, in review)
```

---

## Issue 2 (must fix): Milestone B status contradicts its own notes

**File:** `docs/MILESTONE_A.md:10` and `docs/PROGRESS_REPORT.md:16`

Both files contain an identical table row:

```markdown
| B — KV Cache Correctness | Not started | PR #18 merged |
```

If PR #18 merged work relevant to Milestone B, then the milestone is not "Not started." If PR #18 is actually a Milestone A deliverable (the `llama-kv` crate scaffold) and doesn't count toward Milestone B's correctness validation, the "Notes" column shouldn't reference it under Milestone B since it creates confusion.

**Suggested fix:** Either:
- Remove the PR #18 note from Milestone B's row: `| B — KV Cache Correctness | Not started | |`
- Or clarify: `| B — KV Cache Correctness | Not started | KV scaffold (PR #18) landed in Milestone A; correctness validation not started |`

---

## Issue 3 (should fix): Milestones overview table is duplicated verbatim

The exact same 5-row milestones table appears in both `docs/MILESTONE_A.md:7-13` and `docs/PROGRESS_REPORT.md:13-19`. This creates a maintenance burden — any status update must be applied in two places, and they will inevitably drift apart (as is already happening with the sampling crate status).

**Suggested fix:** Keep the full milestones overview table only in `docs/PROGRESS_REPORT.md` (the canonical progress tracker). In `docs/MILESTONE_A.md`, replace the table with a reference link.

---

## Issue 4 (should fix): MILESTONE_A.md checklist inconsistent with PROGRESS_REPORT.md checklist

The Milestone A checklist in `MILESTONE_A.md:22-29` has 7 items, while `PROGRESS_REPORT.md:23-32` has 8 items (it includes "Architecture + TDD docs | PR #3 (merged)"). These should match, or one should clearly be the authoritative source.

---

## Issue 5 (nit): PR description is empty

The PR title is "merge develop into main" with no body. A brief description of what's new would help reviewers and future readers of the git history.

---

## Issue 6 (nit): New doc links ordering in README

The PROGRESS_REPORT and MILESTONE_A links are inserted between ROADMAP and ARCHITECTURE. Consider grouping foundational docs (ARCHITECTURE, TEST_STRATEGY) separately from ephemeral tracking docs (PROGRESS_REPORT, MILESTONE_A).

---

## Summary

| Severity | Count | Items |
|----------|-------|-------|
| Must fix | 2 | Sampling crate checkbox, Milestone B status contradiction |
| Should fix | 2 | Duplicated milestones table, checklist inconsistency |
| Nit | 2 | Empty PR description, README doc ordering |

**Verdict:** Request changes on the two "must fix" items. The status inconsistencies will mislead anyone reading these docs to gauge project progress.
