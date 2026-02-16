//! # llama-tokenizer
//!
//! Deterministic tokenization for llama.rs.
//!
//! This crate provides:
//! - A `Tokenizer` trait for pluggable tokenization backends
//! - A reference whitespace tokenizer for testing
//! - Loader APIs for HF `tokenizer.json` and SentencePiece assets
//! - Streaming decoding with UTF-8 handling
//! - Chat template support (future)

use std::collections::HashMap;
use std::fs;
use std::path::Path;
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
    #[error("Asset load error: {0}")]
    AssetLoadError(String),
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

    fn append_bytes(&mut self, bytes: &[u8]) -> TokenizerResult<String> {
        self.pending_utf8.extend_from_slice(bytes);
        let mut emitted = String::new();

        while !self.pending_utf8.is_empty() {
            match std::str::from_utf8(&self.pending_utf8) {
                Ok(valid) => {
                    emitted.push_str(valid);
                    self.pending_utf8.clear();
                    break;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to > 0 {
                        let valid = std::str::from_utf8(&self.pending_utf8[..valid_up_to])
                            .map_err(|e| TokenizerError::DecodingError(e.to_string()))?;
                        emitted.push_str(valid);
                        self.pending_utf8.drain(..valid_up_to);
                        if err.error_len().is_none() {
                            break;
                        }
                        continue;
                    }

                    if err.error_len().is_none() {
                        break;
                    }

                    return Err(TokenizerError::DecodingError(format!(
                        "invalid UTF-8 sequence at byte {}",
                        valid_up_to
                    )));
                }
            }
        }

        self.buffer.push_str(&emitted);
        Ok(emitted)
    }

    /// Finalize the decoding stream and validate that no incomplete UTF-8 bytes remain.
    ///
    /// Call this after the last `decode_token` to ensure the stream ended on a
    /// valid UTF-8 boundary. Returns an error if there are pending bytes that
    /// form an incomplete multi-byte sequence.
    pub fn finalize(&self) -> TokenizerResult<()> {
        if self.pending_utf8.is_empty() {
            Ok(())
        } else {
            Err(TokenizerError::DecodingError(
                "incomplete UTF-8 sequence at end of stream".to_string(),
            ))
        }
    }
}

fn parse_byte_piece(piece: &str) -> Option<u8> {
    if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
        u8::from_str_radix(&piece[3..5], 16).ok()
    } else {
        None
    }
}

fn piece_to_bytes(piece: &str, prepend_space: bool) -> Vec<u8> {
    let mut out = Vec::new();
    if prepend_space {
        out.push(b' ');
    }
    if let Some(byte) = parse_byte_piece(piece) {
        out.push(byte);
    } else {
        out.extend_from_slice(piece.as_bytes());
    }
    out
}

fn load_json_tokenizer_vocab(
    contents: &str,
) -> TokenizerResult<(HashMap<i32, String>, Option<i32>)> {
    let parsed: serde_json::Value = serde_json::from_str(contents)
        .map_err(|e| TokenizerError::AssetLoadError(format!("invalid tokenizer.json: {e}")))?;

    let vocab_obj = parsed
        .pointer("/model/vocab")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            TokenizerError::AssetLoadError(
                "tokenizer.json is missing model.vocab object".to_string(),
            )
        })?;

    let mut vocab = HashMap::with_capacity(vocab_obj.len());
    for (piece, id_value) in vocab_obj {
        let raw_id = id_value.as_u64().ok_or_else(|| {
            TokenizerError::AssetLoadError(format!("non-integer ID for piece '{piece}'"))
        })?;
        if raw_id > i32::MAX as u64 {
            return Err(TokenizerError::AssetLoadError(format!(
                "token ID out of range for piece '{piece}': {raw_id}"
            )));
        }
        vocab.insert(raw_id as i32, piece.clone());
    }

    let unk_piece = parsed
        .pointer("/model/unk_token")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("<unk>");
    let unk_id = vocab
        .iter()
        .find_map(|(id, piece)| if piece == unk_piece { Some(*id) } else { None })
        .or_else(|| {
            vocab
                .iter()
                .find_map(|(id, piece)| if piece == "<unk>" { Some(*id) } else { None })
        });

    Ok((vocab, unk_id))
}

fn load_sentencepiece_vocab(contents: &str) -> HashMap<i32, String> {
    let mut vocab = HashMap::new();
    for (i, line) in contents.lines().enumerate() {
        let token = line
            .split('\t')
            .next()
            .unwrap_or_default()
            .trim()
            .to_string();
        if !token.is_empty() {
            vocab.insert(i as i32, token);
        }
    }
    vocab
}

/// Tokenizer loaded from external assets (`tokenizer.json` or SentencePiece).
pub struct LoadedTokenizer {
    vocab: HashMap<i32, String>,
    reverse_vocab: HashMap<String, i32>,
    unknown_id: Option<i32>,
    max_piece_len: usize,
    byte_pieces: [Option<i32>; 256],
}

impl LoadedTokenizer {
    pub fn from_tokenizer_json_path(path: impl AsRef<Path>) -> TokenizerResult<Self> {
        let contents = fs::read_to_string(path.as_ref()).map_err(|e| {
            TokenizerError::AssetLoadError(format!(
                "failed to read tokenizer.json '{}': {e}",
                path.as_ref().display()
            ))
        })?;
        Self::from_tokenizer_json_str(&contents)
    }

    pub fn from_tokenizer_json_str(contents: &str) -> TokenizerResult<Self> {
        let (vocab, unknown_id) = load_json_tokenizer_vocab(contents)?;
        Ok(Self::from_vocab(vocab, unknown_id))
    }

    pub fn from_sentencepiece_files(
        model_path: impl AsRef<Path>,
        vocab_path: impl AsRef<Path>,
    ) -> TokenizerResult<Self> {
        let model_bytes = fs::read(model_path.as_ref()).map_err(|e| {
            TokenizerError::AssetLoadError(format!(
                "failed to read sentencepiece model '{}': {e}",
                model_path.as_ref().display()
            ))
        })?;
        if model_bytes.is_empty() {
            return Err(TokenizerError::AssetLoadError(format!(
                "sentencepiece model '{}' is empty",
                model_path.as_ref().display()
            )));
        }

        let vocab_contents = fs::read_to_string(vocab_path.as_ref()).map_err(|e| {
            TokenizerError::AssetLoadError(format!(
                "failed to read sentencepiece vocab '{}': {e}",
                vocab_path.as_ref().display()
            ))
        })?;
        let vocab = load_sentencepiece_vocab(&vocab_contents);
        let unknown_id = vocab
            .iter()
            .find_map(|(id, piece)| if piece == "<unk>" { Some(*id) } else { None });

        Ok(Self::from_vocab(vocab, unknown_id))
    }

    fn from_vocab(vocab: HashMap<i32, String>, unknown_id: Option<i32>) -> Self {
        let mut reverse_vocab = HashMap::with_capacity(vocab.len());
        let mut max_piece_len = 0;
        let mut byte_pieces = [None; 256];

        for (&id, piece) in &vocab {
            reverse_vocab.insert(piece.clone(), id);
            max_piece_len = max_piece_len.max(piece.len());

            if let Some(byte) = parse_byte_piece(piece) {
                byte_pieces[byte as usize] = Some(id);
            }
        }

        Self {
            vocab,
            reverse_vocab,
            unknown_id,
            max_piece_len,
            byte_pieces,
        }
    }

    fn decode_piece(&self, token: i32) -> TokenizerResult<&str> {
        self.vocab
            .get(&token)
            .map(String::as_str)
            .ok_or(TokenizerError::InvalidToken(token))
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
        let mut state = DecodingState::new();
        for &token in tokens {
            self.decode_token(token, &mut state)?;
        }
        state.finalize()?;
        Ok(state.buffer)
    }

    fn decode_token(&self, token: i32, state: &mut DecodingState) -> TokenizerResult<String> {
        let piece = self.decode_id(token)?;
        let prepend_space = state.emitted_any && parse_byte_piece(&piece).is_none();
        let emitted = state.append_bytes(&piece_to_bytes(&piece, prepend_space))?;
        state.emitted_any = true;
        Ok(emitted)
    }

    fn vocab_size(&self) -> usize {
        self.state.read().map(|s| s.vocab.len()).unwrap_or(0)
    }
}

impl Tokenizer for LoadedTokenizer {
    fn encode(&self, text: &str) -> TokenizerResult<Vec<i32>> {
        let mut out = Vec::new();

        // Greedy longest-match subword encoding over the full input string.
        let len = text.len();
        let mut i = 0;
        while i < len {
            let mut best_id: Option<i32> = None;
            let mut best_len: usize = 0;

            let max_j = (i + self.max_piece_len).min(len);
            let mut j = i + 1;
            while j <= max_j {
                if !text.is_char_boundary(j) {
                    j += 1;
                    continue;
                }

                let piece = &text[i..j];
                if let Some(&id) = self.reverse_vocab.get(piece) {
                    best_id = Some(id);
                    best_len = j - i;
                }

                j += 1;
            }

            if let Some(id) = best_id {
                out.push(id);
                i += best_len;
            } else {
                // If no direct piece match, try byte-fallback before unknown_id.
                let ch = text[i..].chars().next().ok_or_else(|| {
                    TokenizerError::EncodingError("unexpected end of string".to_string())
                })?;
                let ch_len = ch.len_utf8();
                let ch_bytes = &text[i..i + ch_len].as_bytes();

                let mut all_bytes_matched = true;
                let mut byte_ids = Vec::with_capacity(ch_len);
                for &b in *ch_bytes {
                    if let Some(id) = self.byte_pieces[b as usize] {
                        byte_ids.push(id);
                    } else {
                        all_bytes_matched = false;
                        break;
                    }
                }

                if all_bytes_matched {
                    out.extend(byte_ids);
                    i += ch_len;
                } else if let Some(unknown_id) = self.unknown_id {
                    out.push(unknown_id);
                    i += ch_len;
                } else {
                    let snippet: String = text[i..].chars().take(16).collect();
                    return Err(TokenizerError::EncodingError(format!(
                        "input text starting with '{snippet}' missing from loaded vocabulary (no byte fallback)"
                    )));
                }
            }
        }
        Ok(out)
    }

    fn decode(&self, tokens: &[i32]) -> TokenizerResult<String> {
        let mut state = DecodingState::new();
        for &token in tokens {
            self.decode_token(token, &mut state)?;
        }
        state.finalize()?;
        Ok(state.buffer)
    }

    fn decode_token(&self, token: i32, state: &mut DecodingState) -> TokenizerResult<String> {
        let piece = self.decode_piece(token)?;
        // For loaded HF/SentencePiece-style tokenizers, tokens encode whitespace
        // explicitly (via markers or literal spaces). Concatenate pieces as-is.
        let emitted = state.append_bytes(&piece_to_bytes(piece, false))?;
        state.emitted_any = true;
        Ok(emitted)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// RAII temp file that auto-deletes on drop.
    struct TempFile(std::path::PathBuf);

    impl TempFile {
        fn new(name: &str, contents: &[u8]) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!("llama-tokenizer-{name}-{nanos}"));
            fs::write(&path, contents).unwrap();
            Self(path)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

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
        assert_eq!(
            tok.decode(&[999]).unwrap_err(),
            TokenizerError::InvalidToken(999)
        );
    }

    #[test]
    fn roundtrip_unicode_words() {
        let tok = WhitespaceTokenizer::new();
        // Unicode words separated by spaces round-trip correctly
        let original = "café naïve résumé";
        let encoded = tok.encode(original).unwrap();
        let decoded = tok.decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn roundtrip_emoji() {
        let tok = WhitespaceTokenizer::new();
        let original = "hello 🦀 world 🚀";
        let encoded = tok.encode(original).unwrap();
        assert_eq!(encoded.len(), 4);
        let decoded = tok.decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn roundtrip_cjk() {
        let tok = WhitespaceTokenizer::new();
        // CJK characters as whitespace-delimited tokens
        let original = "你好 世界";
        let encoded = tok.encode(original).unwrap();
        assert_eq!(encoded.len(), 2);
        let decoded = tok.decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_tabs_and_newlines_treated_as_whitespace() {
        let tok = WhitespaceTokenizer::new();
        // split_whitespace treats tabs and newlines as delimiters
        let ids = tok.encode("hello\tworld\ntest").unwrap();
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn deterministic_encoding() {
        // Same input always produces same token IDs
        let tok = WhitespaceTokenizer::new();
        let ids1 = tok.encode("hello world test").unwrap();
        let ids2 = tok.encode("hello world test").unwrap();
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn streaming_decode_multiple_tokens() {
        let tok: &dyn Tokenizer = &WhitespaceTokenizer::new();
        let encoded = tok.encode("the quick brown fox").unwrap();
        let mut state = DecodingState::new();

        let mut accumulated = String::new();
        for &id in &encoded {
            let chunk = tok.decode_token(id, &mut state).unwrap();
            accumulated.push_str(&chunk);
        }

        assert_eq!(accumulated, "the quick brown fox");
        assert_eq!(state.buffer(), "the quick brown fox");
    }

    #[test]
    fn vocab_size_reflects_built_vocab() {
        let tok = WhitespaceTokenizer::new();
        assert_eq!(tok.vocab_size(), 0);
        tok.encode("hello world hello").unwrap();
        assert_eq!(tok.vocab_size(), 2);
    }

    #[test]
    fn loaded_tokenizer_reads_hf_tokenizer_json() {
        let json = r#"{
          "model": {
            "unk_token": "<unk>",
            "vocab": {
              "<unk>": 0,
              "hello": 1,
              " ": 3,
              "world": 2
            }
          }
        }"#;
        let file = TempFile::new("tokenizer.json", json.as_bytes());
        let tok = LoadedTokenizer::from_tokenizer_json_path(file.path()).unwrap();
        assert_eq!(tok.vocab_size(), 4);
        // Greedy longest-match: "hello" → 1, " " → 3, "world" → 2
        assert_eq!(tok.encode("hello world").unwrap(), vec![1, 3, 2]);
    }

    #[test]
    fn loaded_tokenizer_reads_sentencepiece_assets() {
        let model_file = TempFile::new("sp.model", b"fake-model-binary");
        let vocab_file = TempFile::new(
            "sp.vocab",
            "<unk>\t0\nhello\t-1.2\n▁\t-1.5\nworld\t-2.3\n".as_bytes(),
        );

        let tok = LoadedTokenizer::from_sentencepiece_files(model_file.path(), vocab_file.path())
            .unwrap();
        assert_eq!(tok.vocab_size(), 4);
        assert_eq!(tok.encode("hello").unwrap(), vec![1]);
        assert_eq!(tok.encode("world").unwrap(), vec![3]);
        assert_eq!(tok.decode(&[1, 3]).unwrap(), "helloworld");
    }

    #[test]
    fn loaded_tokenizer_roundtrip_edge_cases() {
        let json = r#"{
          "model": {
            "unk_token": "<unk>",
            "vocab": {
              "<unk>": 0,
              "hello": 1,
              "世界": 2,
              "🙂": 3,
              " ": 4
            }
          }
        }"#;
        let tok = LoadedTokenizer::from_tokenizer_json_str(json).unwrap();
        let input = "hello 世界 🙂";
        let encoded = tok.encode(input).unwrap();
        assert_eq!(encoded, vec![1, 4, 2, 4, 3]);
        let decoded = tok.decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn streaming_decode_handles_utf8_boundaries() {
        let json = r#"{
          "model": {
            "unk_token": "<unk>",
            "vocab": {
              "<unk>": 0,
              "<0xF0>": 1,
              "<0x9F>": 2,
              "<0x99>": 3,
              "<0x82>": 4
            }
          }
        }"#;
        let tok: &dyn Tokenizer = &LoadedTokenizer::from_tokenizer_json_str(json).unwrap();
        let mut state = DecodingState::new();

        assert_eq!(tok.decode_token(1, &mut state).unwrap(), "");
        assert_eq!(tok.decode_token(2, &mut state).unwrap(), "");
        assert_eq!(tok.decode_token(3, &mut state).unwrap(), "");
        assert_eq!(tok.decode_token(4, &mut state).unwrap(), "🙂");
        assert_eq!(state.buffer(), "🙂");
    }

    #[test]
    fn streaming_decode_rejects_invalid_utf8() {
        let json = r#"{
          "model": {
            "unk_token": "<unk>",
            "vocab": {
              "<unk>": 0,
              "<0xFF>": 1
            }
          }
        }"#;
        let tok: &dyn Tokenizer = &LoadedTokenizer::from_tokenizer_json_str(json).unwrap();
        let mut state = DecodingState::new();
        let err = tok.decode_token(1, &mut state).unwrap_err();
        assert!(matches!(err, TokenizerError::DecodingError(_)));
    }

    #[test]
    fn loaded_tokenizer_byte_pieces_encode() {
        let json = r#"{
          "model": {
            "unk_token": "<unk>",
            "vocab": {
              "<unk>": 0,
              "<0x41>": 1
            }
          }
        }"#;
        let tok = LoadedTokenizer::from_tokenizer_json_str(json).unwrap();
        // 'A' is 0x41. It should match "<0x41>" if we handle byte pieces.
        // Currently it fails because "A" != "<0x41>".
        let encoded = tok.encode("A").unwrap();
        assert_eq!(encoded, vec![1]);
    }
}
