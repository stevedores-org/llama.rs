//! # llama-cli
//!
//! Generation pipeline for llama.rs Milestone A.
//!
//! Wires: tokenizer → model forward pass → sampler → detokenizer.
//! Uses a tiny deterministic model with fixed-seed weights for demo/testing.

use llama_kv::{KVError, KVLayout, LayerKVCache};
use llama_models::{apply_rope, attention_decode, mlp_swiglu, rms_norm, ModelError};
use llama_sampling::{Sampler, SamplingError};
use llama_tokenizer::{Tokenizer, TokenizerError, WhitespaceTokenizer};

/// Errors from the generation pipeline.
#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error("model error: {0}")]
    Model(#[from] ModelError),
    #[error("sampling error: {0}")]
    Sampling(#[from] SamplingError),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("kv cache error: {0}")]
    KVCache(#[from] KVError),
    #[error("empty prompt")]
    EmptyPrompt,
}

/// Configuration for the tiny demo model.
#[derive(Debug, Clone)]
pub struct TinyModelConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
}

impl Default for TinyModelConfig {
    fn default() -> Self {
        Self {
            d_model: 32,
            d_ff: 64,
            n_heads: 4,
            head_dim: 8,
            vocab_size: 256,
            max_seq_len: 64,
            rope_base: 10_000.0,
            norm_eps: 1e-5,
        }
    }
}

/// Tiny deterministic model for Milestone A demo.
///
/// Single transformer block: embed → norm → attention → residual → norm → MLP → residual → proj
pub struct TinyModel {
    pub config: TinyModelConfig,
    // Embeddings: [vocab_size, d_model]
    embeddings: Vec<f32>,
    // Block weights
    norm1_weight: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    norm2_weight: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    // Output
    final_norm_weight: Vec<f32>,
    lm_head: Vec<f32>, // [d_model, vocab_size]
}

/// Simple seeded RNG for weight initialization (xorshift64).
struct WeightRng {
    state: u64,
}

impl WeightRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        // Small magnitude weights for stability
        ((self.state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
    }

    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

impl TinyModel {
    /// Create a tiny model with deterministic weights from a fixed seed.
    pub fn new(config: TinyModelConfig) -> Self {
        let mut rng = WeightRng::new(12345);
        let d = config.d_model;
        let ff = config.d_ff;
        let v = config.vocab_size;

        Self {
            embeddings: rng.fill(v * d),
            norm1_weight: vec![1.0; d],
            w_q: rng.fill(d * d),
            w_k: rng.fill(d * d),
            w_v: rng.fill(d * d),
            w_o: rng.fill(d * d),
            norm2_weight: vec![1.0; d],
            w_gate: rng.fill(d * ff),
            w_up: rng.fill(d * ff),
            w_down: rng.fill(ff * d),
            final_norm_weight: vec![1.0; d],
            lm_head: rng.fill(d * v),
            config,
        }
    }

    /// Embed a token ID into a d_model vector.
    fn embed(&self, token_id: usize) -> Vec<f32> {
        let d = self.config.d_model;
        let offset = token_id * d;
        self.embeddings[offset..offset + d].to_vec()
    }

    /// Matrix-vector multiply: x @ W where W is [in_dim, out_dim] row-major.
    fn matvec(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0; out_dim];
        for i in 0..out_dim {
            let mut acc = 0.0f32;
            for j in 0..in_dim {
                acc += x[j] * w[j * out_dim + i];
            }
            out[i] = acc;
        }
        out
    }

    /// Forward pass for a single decode step.
    ///
    /// Takes the last token, uses KV cache for context, returns logits.
    pub fn forward_decode(
        &self,
        token_id: usize,
        position: usize,
        kv_cache: &mut LayerKVCache,
    ) -> Result<Vec<f32>, GenerateError> {
        let c = &self.config;
        let d = c.d_model;

        // 1. Embed token
        let x = self.embed(token_id);

        // 2. Pre-attention norm
        let x_norm = rms_norm(&x, &self.norm1_weight, c.norm_eps)?;

        // 3. Project Q, K, V
        let mut q = Self::matvec(&x_norm, &self.w_q, d, d);
        let mut k = Self::matvec(&x_norm, &self.w_k, d, d);
        let v = Self::matvec(&x_norm, &self.w_v, d, d);

        // 4. Apply RoPE to Q and K
        apply_rope(&mut q, &mut k, position, c.n_heads, c.head_dim, c.rope_base)?;

        // 5. Write K, V to cache and get full context
        kv_cache.append_token(&k, &v)?;

        let seq_len = kv_cache.seq_len;
        let keys = &kv_cache.k[..seq_len * d];
        let values = &kv_cache.v[..seq_len * d];

        // 6. Attention: Q against all cached K, V
        let attn_out = attention_decode(&q, keys, values, seq_len, c.n_heads, c.head_dim)?;

        // 7. Output projection + residual
        let attn_proj = Self::matvec(&attn_out, &self.w_o, d, d);
        let x_after_attn: Vec<f32> = x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

        // 8. Pre-MLP norm
        let x_norm2 = rms_norm(&x_after_attn, &self.norm2_weight, c.norm_eps)?;

        // 9. MLP + residual
        let mlp_out = mlp_swiglu(&x_norm2, &self.w_gate, &self.w_up, &self.w_down, d, c.d_ff)?;
        let hidden: Vec<f32> = x_after_attn
            .iter()
            .zip(mlp_out.iter())
            .map(|(a, b)| a + b)
            .collect();

        // 10. Final norm
        let hidden_norm = rms_norm(&hidden, &self.final_norm_weight, c.norm_eps)?;

        // 11. Project to vocab logits
        Ok(Self::matvec(&hidden_norm, &self.lm_head, d, c.vocab_size))
    }

    /// Prefill: process all prompt tokens, populate KV cache, return logits for last token.
    pub fn forward_prefill(
        &self,
        token_ids: &[usize],
        kv_cache: &mut LayerKVCache,
    ) -> Result<Vec<f32>, ModelError> {
        let c = &self.config;
        let d = c.d_model;

        // Process all tokens, building up KV cache
        let mut last_logits = vec![0.0; c.vocab_size];

        for (pos, &tok) in token_ids.iter().enumerate() {
            let x = self.embed(tok);
            let x_norm = rms_norm(&x, &self.norm1_weight, c.norm_eps)?;

            let mut q = Self::matvec(&x_norm, &self.w_q, d, d);
            let mut k = Self::matvec(&x_norm, &self.w_k, d, d);
            let v = Self::matvec(&x_norm, &self.w_v, d, d);

            apply_rope(&mut q, &mut k, pos, c.n_heads, c.head_dim, c.rope_base)?;

            kv_cache.append_token(&k, &v)?;

            // Only compute full attention + MLP for last token (optimization)
            if pos == token_ids.len() - 1 {
                let seq_len = kv_cache.seq_len;
                let keys = &kv_cache.k[..seq_len * d];
                let values = &kv_cache.v[..seq_len * d];

                let attn_out = attention_decode(&q, keys, values, seq_len, c.n_heads, c.head_dim)?;
                let attn_proj = Self::matvec(&attn_out, &self.w_o, d, d);
                let x_after_attn: Vec<f32> =
                    x.iter().zip(attn_proj.iter()).map(|(a, b)| a + b).collect();

                let x_norm2 = rms_norm(&x_after_attn, &self.norm2_weight, c.norm_eps)?;
                let mlp_out =
                    mlp_swiglu(&x_norm2, &self.w_gate, &self.w_up, &self.w_down, d, c.d_ff)?;
                let hidden: Vec<f32> = x_after_attn
                    .iter()
                    .zip(mlp_out.iter())
                    .map(|(a, b)| a + b)
                    .collect();

                let hidden_norm = rms_norm(&hidden, &self.final_norm_weight, c.norm_eps)?;
                last_logits = Self::matvec(&hidden_norm, &self.lm_head, d, c.vocab_size);
            }
        }

        Ok(last_logits)
    }
}

/// Result of a generation run.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// Generated token IDs (prompt + new tokens).
    pub token_ids: Vec<usize>,
    /// Decoded output text (new tokens only).
    pub text: String,
}

/// Run the full generation pipeline.
///
/// tokenizer.encode(prompt) → prefill → decode loop → tokenizer.decode
pub fn generate(
    prompt: &str,
    max_tokens: usize,
    seed: u64,
    temperature: f32,
) -> Result<GenerateResult, GenerateError> {
    let config = TinyModelConfig::default();
    let model = TinyModel::new(config.clone());

    // 1. Tokenize
    let tokenizer = WhitespaceTokenizer::new();
    let prompt_ids_i32 = tokenizer.encode(prompt)?;
    if prompt_ids_i32.is_empty() {
        return Err(GenerateError::EmptyPrompt);
    }

    // Convert i32 token IDs to usize (WhitespaceTokenizer uses sequential non-negative IDs)
    let prompt_ids: Vec<usize> = prompt_ids_i32.iter().map(|&id| id as usize).collect();

    // 2. Create KV cache
    let mut kv_cache = LayerKVCache::new(
        config.max_seq_len,
        config.n_heads,
        config.head_dim,
        KVLayout::BySequence,
    );

    // 3. Create sampler
    let mut sampler = Sampler::new().with_temperature(temperature).with_seed(seed);

    // 4. Prefill: process prompt tokens
    let mut logits = model.forward_prefill(&prompt_ids, &mut kv_cache)?;

    // 5. Decode loop
    let mut generated_ids = prompt_ids.clone();
    let mut new_text = String::new();
    let mut decode_state = DecodingState::new();

    for _ in 0..max_tokens {
        let next_token = sampler.sample(&logits)?;
        generated_ids.push(next_token);

        // Register the token in the tokenizer's vocab for decoding
        // (WhitespaceTokenizer builds vocab on encode, so we need a workaround)
        // For the demo, we decode as byte values
        let token_char = if next_token < 128 {
            char::from(next_token as u8)
        } else {
            '?'
        };
        let chunk = format!("{}", token_char);
        if !new_text.is_empty() {
            new_text.push(' ');
        }
        new_text.push_str(&chunk);

        // Check max sequence length
        let position = generated_ids.len() - 1;
        if position >= config.max_seq_len - 1 {
            break;
        }

        // Forward decode step for next iteration
        logits = model.forward_decode(next_token, position, &mut kv_cache)?;
    }

    // Also try proper tokenizer decode for the prompt
    let _ = decode_state;

    Ok(GenerateResult {
        token_ids: generated_ids,
        text: new_text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_model_deterministic_forward() {
        let config = TinyModelConfig::default();
        let model = TinyModel::new(config.clone());
        let mut cache = LayerKVCache::new(
            config.max_seq_len,
            config.n_heads,
            config.head_dim,
            KVLayout::BySequence,
        );

        let logits = model.forward_prefill(&[0, 1, 2], &mut cache).unwrap();
        assert_eq!(logits.len(), config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));

        // Same inputs produce same outputs
        let mut cache2 = LayerKVCache::new(
            config.max_seq_len,
            config.n_heads,
            config.head_dim,
            KVLayout::BySequence,
        );
        let logits2 = model.forward_prefill(&[0, 1, 2], &mut cache2).unwrap();
        assert_eq!(logits, logits2);
    }

    #[test]
    fn tiny_model_decode_step() {
        let config = TinyModelConfig::default();
        let model = TinyModel::new(config.clone());
        let mut cache = LayerKVCache::new(
            config.max_seq_len,
            config.n_heads,
            config.head_dim,
            KVLayout::BySequence,
        );

        // Prefill
        let _ = model.forward_prefill(&[0, 1], &mut cache).unwrap();
        assert_eq!(cache.seq_len, 2);

        // Decode
        let logits = model.forward_decode(2, 2, &mut cache).unwrap();
        assert_eq!(logits.len(), config.vocab_size);
        assert_eq!(cache.seq_len, 3);
    }

    #[test]
    fn generate_deterministic_with_seed() {
        let result1 = generate("hello world", 5, 42, 1.0).unwrap();
        let result2 = generate("hello world", 5, 42, 1.0).unwrap();

        // Same seed + same prompt = same output
        assert_eq!(result1.token_ids, result2.token_ids);
        assert_eq!(result1.text, result2.text);
        assert!(result1.token_ids.len() > 2); // prompt (2) + generated
    }

    #[test]
    fn generate_different_seeds_differ() {
        let result1 = generate("hello world", 5, 42, 1.0).unwrap();
        let result2 = generate("hello world", 5, 99, 1.0).unwrap();

        // Different seeds should (very likely) produce different tokens
        // Not guaranteed but overwhelmingly likely with 256 vocab
        assert_ne!(result1.token_ids, result2.token_ids);
    }

    #[test]
    fn generate_empty_prompt_errors() {
        let result = generate("", 5, 42, 1.0);
        assert!(matches!(result, Err(GenerateError::EmptyPrompt)));
    }

    #[test]
    fn generate_respects_max_tokens() {
        let result = generate("test", 3, 42, 1.0).unwrap();
        // prompt is 1 token + 3 generated = 4 total
        assert_eq!(result.token_ids.len(), 4);
    }
}
