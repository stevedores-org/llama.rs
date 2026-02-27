use llama_tokenizer::{Tokenizer, WhitespaceTokenizer};

#[test]
fn test_whitespace_tokenizer_newline_bug() {
    let tok = WhitespaceTokenizer::new();
    let original = "hello\nworld";
    let encoded = tok.encode(original).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    println!("Original: {:?}, Decoded: {:?}", original, decoded);
    assert_eq!(decoded, original);
}
