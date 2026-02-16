//! # llama-runtime
//!
//! Runtime backend selection and execution for llama.rs. Integrates with oxidizedMLX
//! for CPU/Metal backends, manages tensor allocation, kernel availability matrix,
//! and telemetry hooks (TTFT, tok/s, memory).

use llama_engine::{
    LlamaEngine, LlamaError, ModelHandle, ModelSpec, PrefillResult, Result, Session, TokenId,
    TokenStream,
};
use llama_tokenizer::{Tokenizer, WhitespaceTokenizer};
use llama_sampling::Sampler;

/// A mock engine implementation for Milestone A.
///
/// Uses a simple whitespace tokenizer and greedy sampler to demonstrate
/// the "narrow waist" API without requiring a real model or MLX backend.
pub struct MockEngine {
    tokenizer: WhitespaceTokenizer,
    sampler: Sampler,
}

impl MockEngine {
    /// Create a new mock engine.
    pub fn new() -> Self {
        Self {
            tokenizer: WhitespaceTokenizer::new(),
            sampler: Sampler::new(),
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
        // Mock loading
        Ok(ModelHandle)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<TokenId>> {
        self.tokenizer
            .encode(text)
            .map_err(|e| LlamaError::Tokenization(e.to_string()))
    }

    fn detokenize(&self, tokens: &[TokenId]) -> Result<String> {
        self.tokenizer
            .decode(tokens)
            .map_err(|e| LlamaError::Tokenization(e.to_string()))
    }

    fn prefill(&self, _session: &mut Session, tokens: &[TokenId]) -> Result<PrefillResult> {
        // In a real engine, this would populate the KV cache
        Ok(PrefillResult {
            tokens_processed: tokens.len(),
        })
    }

    fn decode(&self, _session: &mut Session) -> Result<TokenStream> {
        // Mock generation of a few tokens
        // In a real engine, this would run the model and sampler in a loop
        let mock_logits = vec![0.1, 0.5, 0.1, 0.1, 0.2];
        let token = self.sampler
            .sample(&mock_logits)
            .map_err(|e| LlamaError::Inference(format!("{:?}", e)))? as TokenId;

        Ok(TokenStream::new(vec![token]))
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; 128]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_engine_flow() {
        let engine = MockEngine::new();
        let mut session = Session::new();

        let tokens = engine.tokenize("hello world").unwrap();
        assert_eq!(tokens.len(), 2);

        let prefill = engine.prefill(&mut session, &tokens).unwrap();
        assert_eq!(prefill.tokens_processed, 2);

        let mut stream = engine.decode(&mut session).unwrap();
        let next_token = stream.next().unwrap().unwrap();
        assert!(next_token >= 0);
    }
}
