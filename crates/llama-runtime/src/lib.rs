//! # llama-runtime
//!
//! Runtime backend selection and execution for llama.rs. Integrates with oxidizedMLX
//! for CPU/Metal backends, manages tensor allocation, kernel availability matrix,
//! and telemetry hooks (TTFT, tok/s, memory).
//!
//! For Milestone A, provides a `MockEngine` that demonstrates the narrow-waist
//! `LlamaEngine` trait using whitespace tokenization and greedy sampling.

use llama_engine::{
    LlamaEngine, LlamaError, ModelHandle, ModelSpec, PrefillResult, Result, Session, TokenStream,
};
use llama_sampling::Sampler;
use llama_tokenizer::{Tokenizer, WhitespaceTokenizer};
use std::sync::Mutex;

/// A mock engine implementation for Milestone A.
///
/// Uses a simple whitespace tokenizer and greedy sampler to demonstrate
/// the "narrow waist" API without requiring a real model or MLX backend.
pub struct MockEngine {
    tokenizer: WhitespaceTokenizer,
    sampler: Mutex<Sampler>,
}

impl MockEngine {
    pub fn new() -> Self {
        Self {
            tokenizer: WhitespaceTokenizer::new(),
            sampler: Mutex::new(Sampler::new()),
        }
    }
}

impl Default for MockEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaEngine for MockEngine {
    fn load_model(&self, _spec: &ModelSpec) -> Result<ModelHandle> {
        Ok(ModelHandle { id: 1 })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenizer
            .encode(text)
            .map_err(|e| LlamaError::Tokenization(e.to_string()))
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.tokenizer
            .decode(tokens)
            .map_err(|e| LlamaError::Tokenization(e.to_string()))
    }

    fn prefill(&self, _session: &mut Session, _tokens: &[i32]) -> Result<PrefillResult> {
        Ok(PrefillResult)
    }

    fn decode(&self, _session: &mut Session) -> Result<TokenStream> {
        // Mock: sample from uniform logits to produce a token
        let mock_logits = vec![0.1, 0.5, 0.1, 0.1, 0.2];
        let mut sampler = self.sampler.lock().unwrap();
        let _token = sampler
            .sample(&mock_logits)
            .map_err(|e| LlamaError::Inference(format!("{}", e)))?;

        Ok(TokenStream)
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; 128]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_engine_tokenize_roundtrip() {
        let engine = MockEngine::new();
        let tokens = engine.tokenize("hello world").unwrap();
        assert_eq!(tokens.len(), 2);

        let text = engine.detokenize(&tokens).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn mock_engine_prefill_decode() {
        let engine = MockEngine::new();
        let mut session = Session { id: 1 };

        let tokens = engine.tokenize("hello world").unwrap();
        let _prefill = engine.prefill(&mut session, &tokens).unwrap();
        let _stream = engine.decode(&mut session).unwrap();
    }

    #[test]
    fn mock_engine_embed() {
        let engine = MockEngine::new();
        let embeddings = engine.embed(&["hello", "world"]).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 128);
    }

    #[test]
    fn mock_engine_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockEngine>();
    }
}
