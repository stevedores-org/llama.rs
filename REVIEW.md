# Code Review: PR #17 — Tokenizer trait and reference implementation

## Summary

The PR restructures the project from a single crate to a 6-crate workspace, adds CI, developer tooling (justfile), documentation, and implements the foundational `Tokenizer` trait with a `WhitespaceTokenizer` reference implementation. The workspace scaffolding, CI setup, and `llama-engine` trait design are solid. However, the `WhitespaceTokenizer` implementation has several correctness bugs that undermine the stated goals of deterministic, bidirectional tokenization.

**Verdict: Request Changes** — the workspace scaffold and engine trait are good to merge, but the tokenizer implementation needs fixes before it can serve as a reliable reference for golden tests.

---

## Critical Issues

### 1. `encode()` and `decode()` ignore `self` state — tokenizer is stateless by accident

**Files:** `crates/llama-tokenizer/src/lib.rs:111-141`

Both `encode()` and `decode()` create a **new** `WhitespaceTokenizer` internally and never read from `self.vocab` or `self.reverse_vocab`. The instance fields are dead code.

```rust
// encode() — line 111
fn encode(&self, text: &str) -> TokenizerResult<Vec<i32>> {
    let mut tokenizer = WhitespaceTokenizer::new();  // ← ignores self
    tokenizer.ensure_vocab(text);
    // ... then doesn't even use tokenizer.reverse_vocab, just returns positional indices
}
```

`encode()` maps every word to its positional index (0, 1, 2, ...) regardless of vocabulary. The word "hello" gets ID 0 in `encode("hello world")` but also ID 0 in `encode("hello foo")`. This makes token IDs context-dependent and non-deterministic across calls, which directly contradicts the crate's stated goal of "deterministic tokenization."

`decode()` has the same problem — it creates a new tokenizer and fills it with synthetic `"word_0"`, `"word_1"` strings, so `decode(encode("hello world"))` returns `"word_0 word_1"`, not `"hello world"`.

**Suggestion:** Either make `WhitespaceTokenizer` truly stateful (require building vocab before use, use interior mutability, or take `&mut self`), or make it stateless by design with a fixed vocabulary scheme (e.g., hash-based token IDs).

### 2. Roundtrip test is a false positive

**File:** `crates/llama-tokenizer/src/lib.rs:187-197`

```rust
fn decode_roundtrip() {
    let original = "hello world test";
    let encoded = tok.encode(original).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    // Only checks word COUNT, not content:
    assert_eq!(original_words.len(), decoded_words.len());
}
```

This test passes because the word count happens to match, but the actual decoded text is `"word_0 word_1 word_2"` — not `"hello world test"`. A true roundtrip test should assert `decoded == original` (or at minimum check word equality). As written, this test gives false confidence that encode/decode are inverses.

### 3. `decode_token()` ignores the token parameter

**File:** `crates/llama-tokenizer/src/lib.rs:143-147`

```rust
fn decode_token(&self, _token: i32, state: &mut DecodingState) -> TokenizerResult<String> {
    state.buffer.push(' ');
    Ok(state.buffer.clone())
}
```

The token ID is discarded (`_token`). Every call appends a space regardless of input. This means streaming decode will produce only spaces. The `streaming_decode_state` test (line 207-213) never calls `decode_token`, so this behavior is untested.

### 4. `vocab_size()` returns hardcoded 1000

**File:** `crates/llama-tokenizer/src/lib.rs:149-151`

Should return `self.vocab.len()` (or at least document why it's hardcoded). A caller relying on this to bounds-check token IDs will get incorrect results.

---

## Design Issues

### 5. `Tokenizer` trait takes `&self` but vocabulary building needs `&mut self`

The `Tokenizer` trait requires `&self` for all methods, but `WhitespaceTokenizer::ensure_vocab` takes `&mut self`. This tension caused the workaround of creating throwaway tokenizers inside each method. Options:
- Use `&mut self` on the trait (breaks `Send + Sync` ergonomics)
- Use interior mutability (`RwLock<HashMap>`) for the vocab
- Pre-build vocabulary at construction time and make `encode`/`decode` purely lookup-based

The third option is cleanest for a reference tokenizer.

### 6. Dual error types without integration

`llama-engine` defines `LlamaError::Tokenization(String)` and `llama-tokenizer` defines `TokenizerError`. There's no `From<TokenizerError> for LlamaError` conversion, and `llama-tokenizer` doesn't depend on `llama-engine`. This will cause friction when wiring things together. Consider either:
- Having `llama-tokenizer` depend on `llama-engine` and use its error type, or
- Adding a `From` impl in the engine crate (with tokenizer as a dependency), or
- Deciding that the tokenizer is intentionally standalone and documenting the integration pattern

### 7. Inconsistent naming: `tokenize`/`detokenize` vs `encode`/`decode`

`LlamaEngine` uses `tokenize()`/`detokenize()` while `Tokenizer` uses `encode()`/`decode()`. Pick one convention. `encode`/`decode` is more standard in the tokenizer domain (matches HuggingFace, tiktoken, etc.).

### 8. `DecodingState` doesn't handle partial UTF-8

The doc claims "streaming UTF-8 character decoding" but `DecodingState` only holds a `String`. For actual partial UTF-8 handling (e.g., a multi-byte character split across two tokens), you'd need a `Vec<u8>` byte buffer to accumulate incomplete sequences before converting to `String`.

### 9. `TokenizerError` manually implements `Display`/`Error` instead of using `thiserror`

`thiserror` is already a workspace dependency (used by `llama-engine`) but `llama-tokenizer` hand-rolls `Display` and `Error` impls. Using `thiserror` would be more consistent and less boilerplate.

---

## Minor / Nits

- **CI `fmt` job doesn't cache:** The `fmt` job skips `Swatinem/rust-cache@v2`, which is fine since it's fast, but worth noting for consistency.
- **Scaffold crates use `#![allow(unused)]`:** Acceptable for now, but add a tracking issue / TODO to remove as code lands.
- **`MILESTONE_A.md` checks off "Tokenizer loads + tokenize/detokenize roundtrip tests" as incomplete** (`[ ]`), which is accurate given the roundtrip test bug above.

---

## What looks good

- **Workspace structure** is clean and well-organized. The crate boundaries make sense.
- **`LlamaEngine` trait** is a good narrow-waist design. The method set (load, tokenize, prefill, decode, embed) covers the core use cases.
- **CI pipeline** covers fmt, clippy, and cross-platform testing (Ubuntu + macOS).
- **`justfile`** provides convenient dev commands.
- **`rust-toolchain.toml`** pins the toolchain consistently.
- **Documentation** (`ARCHITECTURE.md`, `MILESTONE_A.md`) clearly communicates the design intent.

---

## Recommended actions before merge

1. Fix `encode()`/`decode()` to actually use `self` vocabulary (or redesign the vocab strategy)
2. Fix `decode_roundtrip` test to assert content equality, not just word count
3. Implement `decode_token()` to actually decode the given token
4. Make `vocab_size()` return the real size
5. Add a test that calls `decode_token()` through the `Tokenizer` trait
