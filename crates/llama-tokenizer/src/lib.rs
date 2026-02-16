//! # llama-tokenizer
//!
//! Deterministic tokenization for llama.rs.
//!
//! This crate provides:
//! - A `Tokenizer` trait for pluggable tokenization backends
//! - A reference whitespace tokenizer for testing
//! - Streaming decoding with UTF-8 handling
//! - Chat template support (future)

use std::collections::HashMap;
use std::sync::RwLock;

/// Error type for tokenization operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TokenizerError {
    #[error("Invalid token ID: {0}")]
    InvalidToken(i32),
    #[error("Encoding error: {0}")]
    EncodingError(String),
    #[error("Decoding error: {0}")]
    DecodingError(String),
}

pub type TokenizerResult<T> = std::result::Result<T, TokenizerError>;

/// Core tokenizer trait. Implementations can be swapped without changing app code.
pub trait Tokenizer: Send + Sync {
    /// Encode text into a sequence of token IDs.
    fn encode(&self, text: &str) -> TokenizerResult<Vec<i32>>;

    /// Decode a complete sequence of tokens into text.
    fn decode(&self, tokens: &[i32]) -> TokenizerResult<String>;

    /// Decode a single token and accumulate with partial UTF-8 state.
    /// For streaming decoding, this allows emitting printable characters immediately.
    fn decode_token(&self, token: i32, state: &mut DecodingState) -> TokenizerResult<String>;

    /// Get vocabulary size.
    fn vocab_size(&self) -> usize;
}

/// Streaming decoding state for handling partial UTF-8 sequences.
#[derive(Debug, Clone, Default)]
pub struct DecodingState {
    buffer: String,
    pending_utf8: Vec<u8>,
    emitted_any: bool,
}

impl DecodingState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.pending_utf8.clear();
        self.emitted_any = false;
    }
}

/// Reference whitespace tokenizer for Milestone A testing.
///
/// - Splits on whitespace
/// - Bidirectional (encode/decode)
/// - Deterministic
/// - Used for golden tests before real tokenizer.json loading
pub struct WhitespaceTokenizer {
    state: RwLock<VocabState>,
}

#[derive(Debug, Default)]
struct VocabState {
    vocab: HashMap<i32, String>,
    reverse_vocab: HashMap<String, i32>,
    next_id: i32,
}

impl WhitespaceTokenizer {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(VocabState::default()),
        }
    }

    fn decode_id(&self, token: i32) -> TokenizerResult<String> {
        let state = self
            .state
            .read()
            .map_err(|_| TokenizerError::DecodingError("tokenizer lock poisoned".to_string()))?;

        state
            .vocab
            .get(&token)
            .cloned()
            .ok_or(TokenizerError::InvalidToken(token))
    }
}

impl Default for WhitespaceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn encode(&self, text: &str) -> TokenizerResult<Vec<i32>> {
        let mut state = self
            .state
            .write()
            .map_err(|_| TokenizerError::EncodingError("tokenizer lock poisoned".to_string()))?;

        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let id = if let Some(id) = state.reverse_vocab.get(word) {
                *id
            } else {
                let id = state.next_id;
                state.next_id += 1;
                state.reverse_vocab.insert(word.to_string(), id);
                state.vocab.insert(id, word.to_string());
                id
            };
            ids.push(id);
        }

        Ok(ids)
    }

    fn decode(&self, tokens: &[i32]) -> TokenizerResult<String> {
        let mut words = Vec::with_capacity(tokens.len());
        for &id in tokens {
            words.push(self.decode_id(id)?);
        }
        Ok(words.join(" "))
    }

    fn decode_token(&self, token: i32, state: &mut DecodingState) -> TokenizerResult<String> {
        let word = self.decode_id(token)?;
        let emitted = if state.emitted_any {
            format!(" {}", word)
        } else {
            word
        };
        state.buffer.push_str(&emitted);
        state.emitted_any = true;
        Ok(emitted)
    }

    fn vocab_size(&self) -> usize {
        self.state.read().map(|s| s.vocab.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_whitespace_simple() {
        let tok = WhitespaceTokenizer::new();
        let ids = tok.encode("hello world").unwrap();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn encode_empty_string() {
        let tok = WhitespaceTokenizer::new();
        let ids = tok.encode("").unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn encode_single_word() {
        let tok = WhitespaceTokenizer::new();
        let ids = tok.encode("hello").unwrap();
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn encode_multiple_spaces() {
        let tok = WhitespaceTokenizer::new();
        let ids = tok.encode("hello    world").unwrap();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn decode_roundtrip() {
        let tok = WhitespaceTokenizer::new();
        let original = "hello world test";
        let encoded = tok.encode(original).unwrap();
        let decoded = tok.decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn decode_empty_tokens() {
        let tok = WhitespaceTokenizer::new();
        let decoded = tok.decode(&[]).unwrap();
        assert_eq!(decoded, "");
    }

    #[test]
    fn streaming_decode_state() {
        let tok: &dyn Tokenizer = &WhitespaceTokenizer::new();
        let encoded = tok.encode("hello world").unwrap();
        let mut state = DecodingState::new();
        assert_eq!(state.buffer(), "");
        assert_eq!(tok.decode_token(encoded[0], &mut state).unwrap(), "hello");
        assert_eq!(state.buffer(), "hello");
        assert_eq!(tok.decode_token(encoded[1], &mut state).unwrap(), " world");
        assert_eq!(state.buffer(), "hello world");

        state.clear();
        assert_eq!(state.buffer(), "");
    }

    #[test]
    fn decode_invalid_token_errors() {
        let tok = WhitespaceTokenizer::new();
        tok.encode("hello").unwrap();
        assert_eq!(tok.decode(&[999]).unwrap_err(), TokenizerError::InvalidToken(999));
    }

    #[test]
    fn vocab_size_reflects_built_vocab() {
        let tok = WhitespaceTokenizer::new();
        assert_eq!(tok.vocab_size(), 0);
        tok.encode("hello world hello").unwrap();
        assert_eq!(tok.vocab_size(), 2);
    }
}
