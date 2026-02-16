# Code Review: PR #16 — Milestone A: Workspace Scaffold and Engine Trait

## Summary

This PR restructures the repository from a single crate into a 6-crate Cargo workspace and establishes the foundational `LlamaEngine` trait. It includes CI, a justfile, architecture docs, and a stub tokenizer with tests.

Overall this is well-structured scaffolding work. The issues below are mostly design-level questions that are worth resolving now before downstream code depends on these types.

---

## Issues to Address

### 1. CI triggers on `main` but this PR targets `develop`

**File:** `.github/workflows/ci.yml`

The workflow only triggers on pushes/PRs to `main`:
```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
```
Since this PR targets `develop` and the project uses a develop-branch workflow, CI won't run for PRs targeting `develop`. Add `develop` to the trigger branches:
```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
```

### 2. `LlamaEngine` trait methods take `&self` but implementations will likely need mutation

**File:** `crates/llama-engine/src/lib.rs`

`prefill` and `decode` take `&self` (immutable reference) plus `&mut Session`:
```rust
fn prefill(&self, session: &mut Session, tokens: &[i32]) -> Result<PrefillResult>;
fn decode(&self, session: &mut Session) -> Result<TokenStream>;
```
Real implementations will likely need internal mutable state (model caches, memory pools, telemetry counters). This forces interior mutability patterns (`Mutex`, `RwLock`, etc.) in every implementation. Consider whether `&mut self` is more ergonomic, or document the rationale for requiring `Send + Sync` + interior mutability.

### 3. Concrete types on a trait interface limit backend swappability

**File:** `crates/llama-engine/src/lib.rs`

`ModelHandle`, `Session`, `PrefillResult`, and `TokenStream` are concrete (empty) structs, but the trait methods use them directly. When different backends need different internal representations, these types won't be swappable. Consider associated types:
```rust
pub trait LlamaEngine: Send + Sync {
    type ModelHandle;
    type Session;
    type PrefillResult;
    type TokenStream;
    // ...
}
```
This aligns better with the stated goal of backends being swappable without changing application code.

### 4. Token type is `i32` — intentional?

**File:** `crates/llama-engine/src/lib.rs`

Token IDs are `i32` (signed), but token IDs are conceptually non-negative. Most tokenizer libraries use `u32`. Was `i32` chosen for FFI compatibility with llama.cpp? At minimum, consider a type alias (`type TokenId = i32;`) so it's easy to change later and self-documenting.

### 5. Tokenizer stub `encode` discards token content

**File:** `crates/llama-tokenizer/src/lib.rs`

The stub `encode` maps words to sequential indices regardless of content:
```rust
pub fn encode(&self, text: &str) -> Vec<i32> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, _)| i as i32)
        .collect()
}
```
"hello" always maps to `0` no matter the word. The `decode` returns a fixed string. This means the roundtrip test (`detokenize(tokenize(x)) == x`) from MILESTONE_A.md cannot pass with this stub. The milestone checklist correctly marks it as unchecked, but a TODO comment in the code linking to the tracking issue would help.

### 6. `Session` has no constructor

**File:** `crates/llama-engine/src/lib.rs`

`Session` is an empty struct with no `new()`, no `Default` impl, and no public fields. Code outside `llama-engine` cannot construct a `Session` to pass to `prefill`/`decode`. Add a constructor or make it constructible.

---

## Minor / Nits

### 7. `#![allow(unused)]` in stub crates

The four stub crates (`llama-kv`, `llama-models`, `llama-runtime`, `llama-sampling`) have `#![allow(unused)]` at the crate root. Reasonable for scaffolding but should be removed when real code lands. Consider a tracking TODO.

### 8. `llama-tokenizer` has no dependency on `llama-engine`

The tokenizer crate defines standalone `encode`/`decode`. The engine trait also defines `tokenize`/`detokenize`. There's no shared type or dependency between them. It's unclear how these connect — will `llama-runtime` wrap a `Tokenizer` internally? A brief note in the architecture doc would help.

### 9. `rust-toolchain.toml` pins `stable` without a specific version

```toml
[toolchain]
channel = "stable"
```
Builds may break on new stable releases if clippy adds new lints. Consider pinning a specific version (e.g., `channel = "1.78"`) or documenting an MSRV.

### 10. No Cargo.lock committed

For applications/binaries, `Cargo.lock` should be committed for reproducible builds. Since this workspace will produce a CLI binary (`llama-cli generate`), consider committing it.

---

## What Looks Good

- Clean workspace layout with consistent `Cargo.toml` structure across crates
- Well-documented architecture rationale in `docs/ARCHITECTURE.md`
- CI covers fmt, clippy, and cross-platform testing (Ubuntu + macOS)
- `justfile` mirrors CI locally — good developer ergonomics
- The tokenizer tests are simple but meaningful for a stub
- Good use of `thiserror` for the error enum
- The milestone checklist honestly marks what's done vs. pending

## Recommendation

Items 2, 3, 4, and 6 are design-level questions that should be resolved or explicitly documented as intentional before merging, since they'll be hard to change once downstream code depends on these types. Item 1 is a straightforward CI fix. The rest are minor.
