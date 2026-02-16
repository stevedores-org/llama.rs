//! # llama-tokenizer
//!
//! Deterministic tokenization for llama.rs. Will support HF `tokenizer.json`
//! and SentencePiece assets, streaming detokenization with partial UTF-8 handling,
//! and chat templates per model family.

/// Stub tokenizer for Milestone A scaffolding.
///
/// Will be replaced with a real implementation that loads HF tokenizer assets.
pub struct Tokenizer;

impl Tokenizer {
    pub fn new() -> Self {
        Self
    }

    /// Encode text into token IDs.
    ///
    /// Stub: simple whitespace tokenization. Real implementation will use
    /// BPE/SentencePiece from tokenizer assets.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        text.split_whitespace()
            .map(|word| {
                let mut hash: i32 = 0;
                for b in word.bytes() {
                    hash = hash.wrapping_add(b as i32);
                }
                hash.abs()
            })
            .collect()
    }

    /// Decode token IDs back into text.
    ///
    /// Stub: returns placeholder. Real implementation will reconstruct text
    /// from vocabulary, handling partial UTF-8 for streaming.
    pub fn decode(&self, tokens: &[i32]) -> String {
        format!("decoded_{}_tokens", tokens.len())
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_is_deterministic() {
        let tok = Tokenizer::new();
        let ids1 = tok.encode("hello world");
        let ids2 = tok.encode("world hello");
        assert_eq!(ids1.len(), 2);
        assert_eq!(ids2.len(), 2);
        assert_eq!(ids1[0], ids2[1]);
        assert_eq!(ids1[1], ids2[0]);
    }

    #[test]
    fn decode_stub_returns_count() {
        let tok = Tokenizer::new();
        assert_eq!(tok.decode(&[0, 1, 2]), "decoded_3_tokens");
    }

    #[test]
    fn empty_input() {
        let tok = Tokenizer::new();
        assert!(tok.encode("").is_empty());
        assert_eq!(tok.decode(&[]), "decoded_0_tokens");
    }
}
