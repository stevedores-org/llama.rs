//! Inference engine for Llama models.
//!
//! Manages the two-phase inference process:
//! - **Prefill**: Processes the full input prompt in parallel, populating the KV cache.
//! - **Decode**: Generates tokens one at a time autoregressively.
//!
//! The engine coordinates the model, KV cache, tokenizer, and sampler.

pub mod actor;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::cache::{self, KvCache};
use crate::error::{LlamaError, Result};
use crate::mlx::{Array, Dtype};
use crate::model::config::ModelConfig;
use crate::model::transformer::LlamaModel;
use crate::sampling::{self, SamplingConfig};

/// Statistics from a generation run.
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,

    /// Number of tokens generated.
    pub generated_tokens: usize,

    /// Time for the prefill phase in milliseconds.
    pub prefill_time_ms: f64,

    /// Time for the decode phase in milliseconds.
    pub decode_time_ms: f64,

    /// Tokens per second during decode.
    pub tokens_per_second: f64,
}

/// The inference engine that owns the model and KV caches.
pub struct InferenceEngine {
    /// The loaded Llama model.
    model: LlamaModel,

    /// KV caches (one per layer).
    caches: Vec<KvCache>,

    /// Model configuration.
    config: ModelConfig,

    /// Current sequence position (for RoPE offset).
    position: usize,

    /// Cancellation flag (shared with UI).
    is_running: Arc<AtomicBool>,
}

impl InferenceEngine {
    /// Create a new inference engine.
    ///
    /// # Arguments
    /// - `model`: The loaded Llama model.
    /// - `config`: Model configuration.
    /// - `max_seq_len`: Maximum context length for KV cache allocation.
    pub fn new(model: LlamaModel, config: ModelConfig, max_seq_len: usize) -> Self {
        let caches = cache::create_caches(
            config.num_hidden_layers,
            1, // batch_size = 1 for interactive
            max_seq_len,
            config.num_key_value_heads,
            config.head_dim,
            Dtype::Float16,
        );

        InferenceEngine {
            model,
            caches,
            config,
            position: 0,
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get a clone of the cancellation flag for the UI to control.
    pub fn running_flag(&self) -> Arc<AtomicBool> {
        self.is_running.clone()
    }

    /// Phase 1: Prefill — process the prompt and populate the KV cache.
    ///
    /// # Arguments
    /// - `token_ids`: Tokenized prompt as `Vec<i32>`.
    ///
    /// # Returns
    /// The logits for the last token (used to sample the first generated token).
    pub fn prefill(&mut self, token_ids: &[i32]) -> Result<Array> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(LlamaError::Inference("empty prompt".into()));
        }

        if seq_len > self.cache_remaining() {
            return Err(LlamaError::Inference(format!(
                "prompt length {seq_len} exceeds cache capacity {}",
                self.cache_remaining()
            )));
        }

        // Create input tensor: [1, seq_len]
        let input = Array::from_slice_i32(token_ids, &[1, seq_len as i32]);

        // Forward pass with offset = 0 (start of sequence)
        let logits = self.model.forward(&input, &mut self.caches, 0);

        // Force evaluation to populate the KV cache
        logits.eval();

        self.position = seq_len;

        Ok(logits)
    }

    /// Phase 2: Decode — generate tokens one at a time.
    ///
    /// This is the autoregressive loop. Each call generates one token.
    ///
    /// # Arguments
    /// - `token_id`: The last generated (or sampled) token ID.
    ///
    /// # Returns
    /// Logits for the next token, shape `[1, 1, vocab_size]`.
    pub fn decode_step(&mut self, token_id: i32) -> Result<Array> {
        if self.position >= self.cache_capacity() {
            return Err(LlamaError::Inference("context length exceeded".into()));
        }

        // Single token input: [1, 1]
        let input = Array::from_slice_i32(&[token_id], &[1, 1]);

        // Forward pass with offset = current position
        let logits = self.model.forward(&input, &mut self.caches, self.position as i32);

        // Force evaluation
        logits.eval();

        self.position += 1;

        Ok(logits)
    }

    /// Run the complete generation pipeline.
    ///
    /// 1. Prefill the prompt
    /// 2. Sample the first token
    /// 3. Decode loop until stop token or max tokens
    ///
    /// # Arguments
    /// - `prompt_tokens`: Tokenized prompt.
    /// - `sampling_config`: Sampling hyperparameters.
    /// - `max_tokens`: Maximum number of tokens to generate.
    /// - `stop_tokens`: Token IDs that signal end of generation.
    /// - `on_token`: Callback invoked with each generated token ID.
    ///
    /// # Returns
    /// Vector of generated token IDs and generation statistics.
    pub fn generate<F>(
        &mut self,
        prompt_tokens: &[i32],
        sampling_config: &SamplingConfig,
        max_tokens: usize,
        stop_tokens: &[i32],
        mut on_token: F,
    ) -> Result<(Vec<i32>, GenerationStats)>
    where
        F: FnMut(i32),
    {
        self.is_running.store(true, Ordering::Release);

        // Phase 1: Prefill
        let prefill_start = std::time::Instant::now();
        let logits = self.prefill(prompt_tokens)?;
        let prefill_time = prefill_start.elapsed();

        // Sample first token
        let token_id = sampling::sample(&logits, sampling_config);
        token_id.eval();
        let mut current_token = token_id.item_i32();

        let mut generated = vec![current_token];
        on_token(current_token);

        // Phase 2: Decode
        let decode_start = std::time::Instant::now();

        for _ in 1..max_tokens {
            // Check cancellation
            if !self.is_running.load(Ordering::Relaxed) {
                break;
            }

            // Check stop tokens
            if stop_tokens.contains(&current_token) {
                break;
            }

            // Generate next token
            let logits = self.decode_step(current_token)?;
            let next_token = sampling::sample(&logits, sampling_config);
            next_token.eval();
            current_token = next_token.item_i32();

            generated.push(current_token);
            on_token(current_token);
        }

        let decode_time = decode_start.elapsed();
        self.is_running.store(false, Ordering::Release);

        let decode_tokens = generated.len().saturating_sub(1).max(1);
        let stats = GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated.len(),
            prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
            decode_time_ms: decode_time.as_secs_f64() * 1000.0,
            tokens_per_second: decode_tokens as f64 / decode_time.as_secs_f64(),
        };

        Ok((generated, stats))
    }

    /// Reset the engine for a new conversation.
    pub fn reset(&mut self) {
        cache::reset_all_caches(&mut self.caches);
        self.position = 0;
    }

    /// Current sequence position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Remaining cache capacity.
    pub fn cache_remaining(&self) -> usize {
        self.caches.first().map_or(0, |c| c.remaining())
    }

    /// Total cache capacity.
    pub fn cache_capacity(&self) -> usize {
        self.caches.first().map_or(0, |c| c.capacity())
    }

    /// Estimated total cache memory in bytes.
    pub fn cache_memory_bytes(&self) -> usize {
        self.caches.iter().map(|c| c.memory_bytes()).sum()
    }

    /// Model configuration reference.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Stop generation from another thread.
    pub fn cancel(&self) {
        self.is_running.store(false, Ordering::Release);
    }

    /// Whether generation is currently running.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }
}
