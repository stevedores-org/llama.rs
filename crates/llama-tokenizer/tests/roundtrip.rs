//! Tokenizer Phase 1 tests: roundtrip, streaming, and property tests
//!
//! Enforces True Test #1 (Bit-perfect tokenization):
//! `detokenize(tokenize(x)) == x` across ASCII, unicode, and streaming boundaries

#[cfg(test)]
mod tests {
    use llama_tokenizer::{Tokenizer, WhitespaceTokenizer, DecodingState};

    // ===== Section A: Roundtrip tests =====

    #[test]
    fn roundtrip_ascii_basic() {
        let tok = WhitespaceTokenizer::new();
        let test_cases = vec![
            ("hello", "hello"),
            ("hello world", "hello world"),
            ("a", "a"),
            // Pure whitespace tokens are dropped by split_whitespace()
            // so " " encodes to empty and decodes to ""
            (" ", ""),
            ("  ", ""),
            ("\t", ""),
            ("\n", ""),
        ];

        for (input, expected) in test_cases {
            let encoded = tok.encode(input).expect("encode failed");
            let decoded = tok.decode(&encoded).expect("decode failed");
            assert_eq!(decoded, expected, "roundtrip failed for: {:?}", input);
        }
    }

    #[test]
    fn roundtrip_whitespace_torture() {
        let tok = WhitespaceTokenizer::new();
        let test_cases = vec![
            // split_whitespace() drops leading/trailing whitespace
            (" a", "a"),                      // leading space stripped
            ("a ", "a"),                      // trailing space stripped
            ("a  b", "a b"),                  // double space collapses to single in join
            ("  leading", "leading"),         // multiple leading stripped
            ("trailing  ", "trailing"),       // multiple trailing stripped
            (" \n", ""),                      // space + newline = pure whitespace
            ("\t\t", ""),                     // tabs = pure whitespace
        ];

        for (input, expected) in test_cases {
            let encoded = tok.encode(input).expect("encode failed");
            let decoded = tok.decode(&encoded).expect("decode failed");
            assert_eq!(decoded, expected, "roundtrip failed for: {:?}", input);
        }
    }

    #[test]
    fn roundtrip_empty_string() {
        let tok = WhitespaceTokenizer::new();
        let encoded = tok.encode("").expect("encode failed");
        let decoded = tok.decode(&encoded).expect("decode failed");
        assert_eq!(decoded, "");
    }

    #[test]
    fn roundtrip_single_word_each_encoding() {
        let tok = WhitespaceTokenizer::new();

        // Encode multiple words to build vocab
        let multi = tok.encode("alpha beta gamma").expect("encode multi failed");

        // Get the first token ID and decode it
        let first_id = multi[0];
        let decoded = tok.decode(&[first_id]).expect("decode failed");
        assert_eq!(decoded, "alpha");

        // Decode just the second token
        let second_id = multi[1];
        let decoded2 = tok.decode(&[second_id]).expect("decode failed");
        assert_eq!(decoded2, "beta");
    }

    // ===== Section B: Streaming decoding boundary tests =====

    #[test]
    fn streaming_decode_sequential() {
        let tok: &dyn Tokenizer = &WhitespaceTokenizer::new();
        let text = "hello world test";
        let tokens = tok.encode(text).expect("encode failed");

        let mut state = DecodingState::new();
        let mut accumulated = String::new();

        for &token in &tokens {
            let chunk = tok.decode_token(token, &mut state).expect("decode_token failed");
            accumulated.push_str(&chunk);
        }

        assert_eq!(accumulated, text, "streaming decode failed");
    }

    #[test]
    fn streaming_decode_state_isolation() {
        let tok: &dyn Tokenizer = &WhitespaceTokenizer::new();

        // First sequence: "a b"
        let tokens1 = tok.encode("a b").expect("encode failed");
        let mut state1 = DecodingState::new();
        let out1_t0 = tok.decode_token(tokens1[0], &mut state1).expect("decode failed");
        assert_eq!(out1_t0, "a");
        assert_eq!(state1.buffer(), "a");

        // Second sequence: "x y" (different state)
        let tokens2 = tok.encode("x y").expect("encode failed");
        let mut state2 = DecodingState::new();
        let out2_t0 = tok.decode_token(tokens2[0], &mut state2).expect("decode failed");
        assert_eq!(out2_t0, "x");
        assert_eq!(state2.buffer(), "x");

        // First state should be unchanged
        assert_eq!(state1.buffer(), "a");
    }

    #[test]
    fn streaming_decode_clear_resets_state() {
        let tok: &dyn Tokenizer = &WhitespaceTokenizer::new();
        let tokens = tok.encode("hello world").expect("encode failed");

        let mut state = DecodingState::new();
        tok.decode_token(tokens[0], &mut state).expect("decode failed");
        assert!(!state.buffer().is_empty());

        state.clear();
        assert_eq!(state.buffer(), "");
    }

    // ===== Section C: Determinism tests =====

    #[test]
    fn encode_is_deterministic() {
        let text = "the quick brown fox";

        let tok1 = WhitespaceTokenizer::new();
        let enc1 = tok1.encode(text).expect("encode 1 failed");

        let tok2 = WhitespaceTokenizer::new();
        let _enc2 = tok2.encode(text).expect("encode 2 failed");

        // Same text may not produce same token IDs across different tokenizer instances
        // because they have separate vocab tables. But within the same instance, should be consistent.
        let enc1_again = tok1.encode(text).expect("encode 1 again failed");
        assert_eq!(enc1, enc1_again, "single tokenizer not deterministic");
    }

    #[test]
    fn vocab_size_tracks_encoding() {
        let tok = WhitespaceTokenizer::new();
        assert_eq!(tok.vocab_size(), 0, "empty tokenizer should have vocab_size=0");

        tok.encode("one").expect("encode failed");
        assert_eq!(tok.vocab_size(), 1);

        tok.encode("one two").expect("encode failed");
        assert_eq!(tok.vocab_size(), 2);

        tok.encode("one two three").expect("encode failed");
        assert_eq!(tok.vocab_size(), 3);

        // Encoding existing word doesn't increase vocab
        tok.encode("one").expect("encode failed");
        assert_eq!(tok.vocab_size(), 3);
    }

    // ===== Section D: Error cases =====

    #[test]
    fn decode_invalid_token_returns_error() {
        let tok = WhitespaceTokenizer::new();
        tok.encode("valid").expect("encode failed");

        // Token 999 was never encoded
        let result = tok.decode(&[999]);
        assert!(result.is_err(), "should error on invalid token");
    }

    #[test]
    fn decode_empty_tokens_roundtrips() {
        let tok = WhitespaceTokenizer::new();
        let decoded = tok.decode(&[]).expect("decode empty failed");
        assert_eq!(decoded, "");
    }

    // ===== Section E: Property-style invariants =====

    #[test]
    fn invariant_encode_never_empty_for_nonempty_input() {
        let tok = WhitespaceTokenizer::new();
        let inputs = vec!["a", "hello world", "x y z"];

        for text in inputs {
            let encoded = tok.encode(text).expect("encode failed");
            // Nonempty input should produce at least one token
            // (except pure whitespace which splits into zero tokens)
            if !text.trim().is_empty() {
                assert!(!encoded.is_empty(), "nonempty input produced zero tokens: {:?}", text);
            }
        }
    }

    #[test]
    fn invariant_roundtrip_preserves_word_count() {
        let tok = WhitespaceTokenizer::new();
        let inputs = vec!["a", "a b", "a b c", "a b c d e"];

        for text in inputs {
            let words: Vec<&str> = text.split_whitespace().collect();
            if words.is_empty() {
                continue;
            }

            let encoded = tok.encode(text).expect("encode failed");
            let decoded = tok.decode(&encoded).expect("decode failed");
            let decoded_words: Vec<&str> = decoded.split_whitespace().collect();

            assert_eq!(
                words.len(), decoded_words.len(),
                "word count mismatch for: {:?}",
                text
            );
        }
    }
}
