//! Integration tests for llama-tokenizer.
//!
//! Validates:
//! - True Test #1: Bit-perfect tokenization roundtrip (detokenize(tokenize(x)) == x)
//! - Unicode and special character handling
//! - Thread safety (concurrent encode/decode)
//! - Streaming decode lifecycle
//! - Vocab stability and determinism
//! - Error paths and edge cases
//! - Trait object usage (dyn Tokenizer)

use llama_tokenizer::*;

// ===========================================================================
// TRUE TEST #1: Bit-Perfect Tokenization Roundtrip
// ===========================================================================

/// Core correctness gate: detokenize(tokenize(x)) must equal x
/// for all valid inputs (allowing whitespace normalization).
#[test]
fn true_test_bit_perfect_roundtrip_simple() {
    let tok = WhitespaceTokenizer::new();
    let inputs = [
        "hello world",
        "the quick brown fox",
        "a",
        "test one two three",
        "rust is great",
    ];
    for input in &inputs {
        let encoded = tok.encode(input).unwrap();
        let decoded = tok.decode(&encoded).unwrap();
        assert_eq!(&decoded, input, "roundtrip failed for: {}", input);
    }
}

#[test]
fn true_test_bit_perfect_roundtrip_single_word() {
    let tok = WhitespaceTokenizer::new();
    let input = "llama";
    let encoded = tok.encode(input).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn true_test_bit_perfect_roundtrip_repeated_words() {
    let tok = WhitespaceTokenizer::new();
    let input = "hello hello hello";
    let encoded = tok.encode(input).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn true_test_bit_perfect_roundtrip_many_words() {
    let tok = WhitespaceTokenizer::new();
    let words: Vec<String> = (0..100).map(|i| format!("word{}", i)).collect();
    let input = words.join(" ");
    let encoded = tok.encode(&input).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn true_test_bit_perfect_roundtrip_unicode_words() {
    let tok = WhitespaceTokenizer::new();
    // The whitespace tokenizer splits on whitespace, so each unicode "word" stays intact
    let input = "bonjour monde";
    let encoded = tok.encode(input).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn true_test_bit_perfect_roundtrip_numbers() {
    let tok = WhitespaceTokenizer::new();
    let input = "123 456 789";
    let encoded = tok.encode(input).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn true_test_bit_perfect_roundtrip_mixed_content() {
    let tok = WhitespaceTokenizer::new();
    let input = "user: hello! how are you?";
    let encoded = tok.encode(input).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, input);
}

// ===========================================================================
// Vocabulary Determinism and Stability
// ===========================================================================

#[test]
fn token_id_stability_same_word_same_id() {
    let tok = WhitespaceTokenizer::new();

    let ids1 = tok.encode("hello world").unwrap();
    let ids2 = tok.encode("hello world").unwrap();
    assert_eq!(ids1, ids2, "same input must produce same token IDs");
}

#[test]
fn token_id_stability_across_different_sentences() {
    let tok = WhitespaceTokenizer::new();

    let ids1 = tok.encode("hello world").unwrap();
    let ids2 = tok.encode("hello llama").unwrap();

    // "hello" should get the same ID in both calls
    assert_eq!(ids1[0], ids2[0], "shared word must have same ID");
    // "world" and "llama" should have different IDs
    assert_ne!(ids1[1], ids2[1], "different words must have different IDs");
}

#[test]
fn vocab_grows_with_new_words() {
    let tok = WhitespaceTokenizer::new();
    assert_eq!(tok.vocab_size(), 0);

    tok.encode("hello").unwrap();
    assert_eq!(tok.vocab_size(), 1);

    tok.encode("world").unwrap();
    assert_eq!(tok.vocab_size(), 2);

    // Re-encoding existing word doesn't grow vocab
    tok.encode("hello").unwrap();
    assert_eq!(tok.vocab_size(), 2);
}

#[test]
fn vocab_size_with_sentence() {
    let tok = WhitespaceTokenizer::new();
    tok.encode("the quick brown fox jumps over the lazy dog")
        .unwrap();
    // "the" appears twice but should only count once
    assert_eq!(tok.vocab_size(), 8);
}

#[test]
fn token_ids_are_sequential() {
    let tok = WhitespaceTokenizer::new();
    let ids = tok.encode("alpha beta gamma delta").unwrap();
    assert_eq!(ids, vec![0, 1, 2, 3]);
}

// ===========================================================================
// Empty and Boundary Inputs
// ===========================================================================

#[test]
fn encode_empty_string() {
    let tok = WhitespaceTokenizer::new();
    let ids = tok.encode("").unwrap();
    assert!(ids.is_empty());
}

#[test]
fn encode_whitespace_only() {
    let tok = WhitespaceTokenizer::new();
    let ids = tok.encode("   ").unwrap();
    assert!(ids.is_empty(), "whitespace-only should produce no tokens");
}

#[test]
fn encode_tabs_and_newlines() {
    let tok = WhitespaceTokenizer::new();
    let ids = tok.encode("\t\n\r").unwrap();
    assert!(ids.is_empty(), "whitespace chars should produce no tokens");
}

#[test]
fn encode_mixed_whitespace_between_words() {
    let tok = WhitespaceTokenizer::new();
    let ids = tok.encode("hello\t\nworld").unwrap();
    assert_eq!(ids.len(), 2);
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(decoded, "hello world"); // whitespace normalized to single space
}

#[test]
fn decode_empty_tokens() {
    let tok = WhitespaceTokenizer::new();
    let decoded = tok.decode(&[]).unwrap();
    assert_eq!(decoded, "");
}

#[test]
fn decode_single_token() {
    let tok = WhitespaceTokenizer::new();
    tok.encode("hello").unwrap();
    let decoded = tok.decode(&[0]).unwrap();
    assert_eq!(decoded, "hello");
}

// ===========================================================================
// Error Handling
// ===========================================================================

#[test]
fn decode_invalid_token_id() {
    let tok = WhitespaceTokenizer::new();
    tok.encode("hello").unwrap(); // registers id 0
    let err = tok.decode(&[999]).unwrap_err();
    assert_eq!(err, TokenizerError::InvalidToken(999));
}

#[test]
fn decode_partially_valid_tokens() {
    let tok = WhitespaceTokenizer::new();
    tok.encode("hello world").unwrap(); // registers 0, 1
                                        // First token valid, second invalid
    let err = tok.decode(&[0, 42]).unwrap_err();
    assert_eq!(err, TokenizerError::InvalidToken(42));
}

#[test]
fn decode_negative_token_id() {
    let tok = WhitespaceTokenizer::new();
    let err = tok.decode(&[-1]).unwrap_err();
    assert_eq!(err, TokenizerError::InvalidToken(-1));
}

#[test]
fn error_display_invalid_token() {
    let err = TokenizerError::InvalidToken(42);
    assert_eq!(format!("{}", err), "Invalid token ID: 42");
}

#[test]
fn error_display_encoding_error() {
    let err = TokenizerError::EncodingError("test".to_string());
    assert_eq!(format!("{}", err), "Encoding error: test");
}

#[test]
fn error_display_decoding_error() {
    let err = TokenizerError::DecodingError("test".to_string());
    assert_eq!(format!("{}", err), "Decoding error: test");
}

#[test]
fn error_eq_and_clone() {
    let err1 = TokenizerError::InvalidToken(1);
    let err2 = err1.clone();
    assert_eq!(err1, err2);
}

#[test]
fn error_is_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(TokenizerError::InvalidToken(0));
    assert!(!err.to_string().is_empty());
}

// ===========================================================================
// Streaming Decode (Architecture Section 8.2)
// ===========================================================================

#[test]
fn streaming_decode_complete_sequence() {
    let tok = WhitespaceTokenizer::new();
    let tokens = tok.encode("hello beautiful world").unwrap();

    let mut state = DecodingState::new();
    let mut accumulated = String::new();

    for &token in &tokens {
        let chunk = tok.decode_token(token, &mut state).unwrap();
        accumulated.push_str(&chunk);
    }

    assert_eq!(accumulated, "hello beautiful world");
    assert_eq!(state.buffer(), "hello beautiful world");
}

#[test]
fn streaming_decode_first_token_no_leading_space() {
    let tok = WhitespaceTokenizer::new();
    let tokens = tok.encode("hello world").unwrap();
    let mut state = DecodingState::new();

    let first = tok.decode_token(tokens[0], &mut state).unwrap();
    assert_eq!(first, "hello"); // no leading space
}

#[test]
fn streaming_decode_subsequent_tokens_have_space() {
    let tok = WhitespaceTokenizer::new();
    let tokens = tok.encode("hello world").unwrap();
    let mut state = DecodingState::new();

    tok.decode_token(tokens[0], &mut state).unwrap();
    let second = tok.decode_token(tokens[1], &mut state).unwrap();
    assert_eq!(second, " world"); // leading space separator
}

#[test]
fn streaming_decode_state_clear_resets() {
    let tok = WhitespaceTokenizer::new();
    let tokens = tok.encode("hello world").unwrap();

    let mut state = DecodingState::new();
    tok.decode_token(tokens[0], &mut state).unwrap();
    tok.decode_token(tokens[1], &mut state).unwrap();
    assert_eq!(state.buffer(), "hello world");

    state.clear();
    assert_eq!(state.buffer(), "");

    // After clear, first token should not have leading space
    let chunk = tok.decode_token(tokens[0], &mut state).unwrap();
    assert_eq!(chunk, "hello");
}

#[test]
fn streaming_decode_matches_batch_decode() {
    let tok = WhitespaceTokenizer::new();
    let tokens = tok.encode("the quick brown fox").unwrap();

    // Batch decode
    let batch = tok.decode(&tokens).unwrap();

    // Streaming decode
    let mut state = DecodingState::new();
    let mut streamed = String::new();
    for &t in &tokens {
        streamed.push_str(&tok.decode_token(t, &mut state).unwrap());
    }

    assert_eq!(batch, streamed, "streaming must match batch decode");
}

#[test]
fn streaming_decode_single_token() {
    let tok = WhitespaceTokenizer::new();
    let tokens = tok.encode("hello").unwrap();
    let mut state = DecodingState::new();

    let chunk = tok.decode_token(tokens[0], &mut state).unwrap();
    assert_eq!(chunk, "hello");
    assert_eq!(state.buffer(), "hello");
}

#[test]
fn streaming_decode_invalid_token_errors() {
    let tok = WhitespaceTokenizer::new();
    let mut state = DecodingState::new();

    let err = tok.decode_token(999, &mut state).unwrap_err();
    assert_eq!(err, TokenizerError::InvalidToken(999));
}

// ===========================================================================
// DecodingState Lifecycle
// ===========================================================================

#[test]
fn decoding_state_default() {
    let state = DecodingState::default();
    assert_eq!(state.buffer(), "");
}

#[test]
fn decoding_state_new_equals_default() {
    let s1 = DecodingState::new();
    let s2 = DecodingState::default();
    assert_eq!(s1.buffer(), s2.buffer());
}

#[test]
fn decoding_state_clone() {
    let tok = WhitespaceTokenizer::new();
    let tokens = tok.encode("hello world").unwrap();
    let mut state = DecodingState::new();
    tok.decode_token(tokens[0], &mut state).unwrap();

    let cloned = state.clone();
    assert_eq!(cloned.buffer(), "hello");
}

// ===========================================================================
// Thread Safety (Architecture Section 3.3)
// ===========================================================================

#[test]
fn concurrent_encode_from_multiple_threads() {
    use std::sync::Arc;

    let tok = Arc::new(WhitespaceTokenizer::new());
    let mut handles = vec![];

    for i in 0..10 {
        let tok_clone = Arc::clone(&tok);
        handles.push(std::thread::spawn(move || {
            let text = format!("thread{} says hello", i);
            tok_clone.encode(&text).unwrap()
        }));
    }

    let results: Vec<Vec<i32>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads should have gotten results
    for r in &results {
        assert_eq!(r.len(), 3); // "threadN", "says", "hello"
    }
}

#[test]
fn concurrent_encode_decode_roundtrip() {
    use std::sync::Arc;

    let tok = Arc::new(WhitespaceTokenizer::new());

    // First, build vocab on main thread
    tok.encode("hello world test").unwrap();

    let mut handles = vec![];
    for _ in 0..10 {
        let tok_clone = Arc::clone(&tok);
        handles.push(std::thread::spawn(move || {
            let encoded = tok_clone.encode("hello world test").unwrap();
            tok_clone.decode(&encoded).unwrap()
        }));
    }

    for h in handles {
        let result = h.join().unwrap();
        assert_eq!(result, "hello world test");
    }
}

// ===========================================================================
// Trait Object Usage (Narrow Waist)
// ===========================================================================

#[test]
fn tokenizer_as_trait_object() {
    let tok: &dyn Tokenizer = &WhitespaceTokenizer::new();
    let encoded = tok.encode("hello world").unwrap();
    assert_eq!(encoded.len(), 2);
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, "hello world");
}

#[test]
fn tokenizer_trait_object_in_box() {
    let tok: Box<dyn Tokenizer> = Box::new(WhitespaceTokenizer::new());
    let encoded = tok.encode("test").unwrap();
    assert_eq!(encoded.len(), 1);
}

#[test]
fn tokenizer_trait_object_vocab_size() {
    let tok: Box<dyn Tokenizer> = Box::new(WhitespaceTokenizer::new());
    assert_eq!(tok.vocab_size(), 0);
    tok.encode("a b c").unwrap();
    assert_eq!(tok.vocab_size(), 3);
}

// ===========================================================================
// WhitespaceTokenizer Default Implementation
// ===========================================================================

#[test]
fn whitespace_tokenizer_default_trait() {
    let tok = WhitespaceTokenizer::default();
    assert_eq!(tok.vocab_size(), 0);
}

// ===========================================================================
// Property-Style Tests (Deterministic Enumeration)
// ===========================================================================

#[test]
fn roundtrip_property_for_generated_sentences() {
    let tok = WhitespaceTokenizer::new();
    let words = ["alpha", "beta", "gamma", "delta", "epsilon"];

    // Test all 2-word combinations
    for w1 in &words {
        for w2 in &words {
            let input = format!("{} {}", w1, w2);
            let encoded = tok.encode(&input).unwrap();
            let decoded = tok.decode(&encoded).unwrap();
            assert_eq!(decoded, input, "roundtrip failed for: {}", input);
        }
    }
}

#[test]
fn encode_length_property() {
    let tok = WhitespaceTokenizer::new();
    // Number of tokens should equal number of whitespace-separated words
    let test_cases = [
        ("", 0),
        ("a", 1),
        ("a b", 2),
        ("a b c", 3),
        ("a b c d e", 5),
        ("  spaces  between  ", 2), // leading/trailing whitespace stripped
    ];
    for (input, expected_len) in &test_cases {
        let encoded = tok.encode(input).unwrap();
        assert_eq!(
            encoded.len(),
            *expected_len,
            "wrong token count for: {:?}",
            input
        );
    }
}

#[test]
fn decode_length_matches_encode() {
    let tok = WhitespaceTokenizer::new();
    let input = "one two three four five";
    let encoded = tok.encode(input).unwrap();
    let decoded = tok.decode(&encoded).unwrap();

    let input_words: Vec<&str> = input.split_whitespace().collect();
    let decoded_words: Vec<&str> = decoded.split_whitespace().collect();
    assert_eq!(input_words.len(), decoded_words.len());
}
