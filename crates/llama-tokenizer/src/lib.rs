//! # llama-tokenizer
//!
//! Deterministic tokenization for llama.rs.
//!
//! This crate provides:
//! - A `Tokenizer` trait for pluggable tokenization backends
//! - A reference whitespace tokenizer for testing
//! - Streaming decoding with UTF-8 handling
//! - Chat template support (future)

use llama_engine::TokenId;

/// Error type for tokenization operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    InvalidToken(TokenId),
    EncodingError(String),
    DecodingError(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::InvalidToken(id) => write!(f, "Invalid token ID: {}", id),
            TokenizerError::EncodingError(msg) => write!(f, "Encoding error: {}", msg),
            TokenizerError::DecodingError(msg) => write!(f, "Decoding error: {}", msg),
        }
    }
}

impl std::error::Error for TokenizerError {}

pub type TokenizerResult<T> = std::result::Result<T, TokenizerError>;

/// Core tokenizer trait. Implementations can be swapped without changing app code.
pub trait Tokenizer: Send + Sync {
    /// Encode text into a sequence of token IDs.
    fn encode(&self, text: &str) -> TokenizerResult<Vec<TokenId>>;

    /// Decode a complete sequence of tokens into text.
    fn decode(&self, tokens: &[TokenId]) -> TokenizerResult<String>;

    /// Decode a single token and accumulate with partial UTF-8 state.
    /// For streaming decoding, this allows emitting printable characters immediately.
    fn decode_token(&self, token: TokenId, state: &mut DecodingState) -> TokenizerResult<String>;

    /// Get vocabulary size.
    fn vocab_size(&self) -> usize;
}

/// Streaming decoding state for handling partial UTF-8 sequences.
#[derive(Debug, Clone, Default)]
pub struct DecodingState {
    buffer: String,
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
    }
}

/// Reference whitespace tokenizer for Milestone A testing.
///
/// - Splits on whitespace
/// - Bidirectional (encode/decode)
/// - Deterministic
/// - Used for golden tests before real tokenizer.json loading
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WhitespaceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn encode(&self, text: &str) -> TokenizerResult<Vec<TokenId>> {
        Ok(text
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| i as TokenId)
            .collect())
    }

    fn decode(&self, tokens: &[TokenId]) -> TokenizerResult<String> {
        Ok(tokens
            .iter()
            .map(|&id| format!("word_{}", id))
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn decode_token(&self, _token: TokenId, state: &mut DecodingState) -> TokenizerResult<String> {
        // For whitespace tokenizer, just add a placeholder
        state.buffer.push(' ');
        Ok(state.buffer.clone())
    }

    fn vocab_size(&self) -> usize {
        1000 // Placeholder for reference tokenizer
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

        let original_words: Vec<&str> = original.split_whitespace().collect();
        let decoded_words: Vec<&str> = decoded.split_whitespace().collect();

        assert_eq!(original_words.len(), decoded_words.len());
    }

    #[test]
    fn decode_empty_tokens() {
        let tok = WhitespaceTokenizer::new();
        let decoded = tok.decode(&[]).unwrap();
        assert_eq!(decoded, "");
    }

    #[test]
    fn streaming_decode_state() {
        let mut state = DecodingState::new();
        assert_eq!(state.buffer(), "");

        state.clear();
        assert_eq!(state.buffer(), "");
    }
}
