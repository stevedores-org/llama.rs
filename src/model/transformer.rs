//! Llama 3 transformer architecture.
//!
//! Composes the full model from embedding, transformer blocks (attention +
//! feed-forward + RMSNorm), and the language model head.

use crate::cache::KvCache;
use crate::mlx::ops;
use crate::mlx::Array;
use crate::model::attention::Attention;
use crate::model::config::ModelConfig;
use crate::model::embedding::Embedding;
use crate::model::linear::AnyLinear;
use crate::model::norm::RmsNorm;

/// A single transformer block (decoder layer).
///
/// Each block consists of:
/// 1. Pre-attention RMSNorm
/// 2. Grouped Query Attention (with RoPE + KV cache)
/// 3. Residual connection
/// 4. Pre-FFN RMSNorm
/// 5. SwiGLU Feed-Forward Network
/// 6. Residual connection
pub struct TransformerBlock {
    /// Pre-attention normalization.
    pub attention_norm: RmsNorm,

    /// Multi-head attention with GQA.
    pub attention: Attention,

    /// Pre-FFN normalization.
    pub ffn_norm: RmsNorm,

    /// FFN gate projection (SwiGLU): [hidden_size] -> [intermediate_size].
    pub gate_proj: AnyLinear,

    /// FFN up projection: [hidden_size] -> [intermediate_size].
    pub up_proj: AnyLinear,

    /// FFN down projection: [intermediate_size] -> [hidden_size].
    pub down_proj: AnyLinear,
}

impl TransformerBlock {
    /// Forward pass through a single transformer block.
    ///
    /// # Arguments
    /// - `x`: Hidden states `[batch, seq_len, hidden_size]`.
    /// - `cache`: KV cache for this layer.
    /// - `offset`: Position offset for RoPE.
    /// - `mask`: Optional causal mask.
    pub fn forward(
        &self,
        x: &Array,
        cache: &mut KvCache,
        offset: i32,
        mask: Option<&Array>,
    ) -> Array {
        // ── Self-Attention with residual ────────────────────────────────
        let residual = x.clone();
        let h = self.attention_norm.forward(x);
        let h = self.attention.forward(&h, cache, offset, mask);
        let h = ops::add(&residual, &h);

        // ── SwiGLU FFN with residual ────────────────────────────────────
        let residual = h.clone();
        let h_norm = self.ffn_norm.forward(&h);

        // SwiGLU: silu(gate(x)) * up(x)
        let gate = self.gate_proj.forward(&h_norm);
        let gate = ops::silu(&gate);
        let up = self.up_proj.forward(&h_norm);
        let h_ffn = ops::multiply(&gate, &up);
        let h_ffn = self.down_proj.forward(&h_ffn);

        ops::add(&residual, &h_ffn)
    }
}

/// The full Llama transformer model.
pub struct LlamaModel {
    /// Model configuration.
    pub config: ModelConfig,

    /// Token embedding layer.
    pub embed_tokens: Embedding,

    /// Transformer decoder layers.
    pub layers: Vec<TransformerBlock>,

    /// Final RMSNorm before the LM head.
    pub norm: RmsNorm,

    /// Language model head: projects hidden states to vocabulary logits.
    /// May share weights with embed_tokens (tied embeddings).
    pub lm_head: AnyLinear,
}

impl LlamaModel {
    /// Run the full model forward pass.
    ///
    /// # Arguments
    /// - `token_ids`: Input token IDs `[batch, seq_len]`.
    /// - `caches`: One KV cache per layer.
    /// - `offset`: Position offset for RoPE (0 during prefill, seq_pos during decode).
    ///
    /// # Returns
    /// Logits tensor of shape `[batch, seq_len, vocab_size]`.
    pub fn forward(
        &self,
        token_ids: &Array,
        caches: &mut [KvCache],
        offset: i32,
    ) -> Array {
        // Embed tokens: [batch, seq_len] -> [batch, seq_len, hidden_size]
        let mut h = self.embed_tokens.forward(token_ids);

        // Build causal mask for prefill (not needed for single-token decode)
        let seq_len = token_ids.shape()[1];
        let mask = if seq_len > 1 {
            Some(Self::build_causal_mask(seq_len, offset))
        } else {
            None
        };

        // Pass through each transformer block
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(&h, cache, offset, mask.as_ref());
        }

        // Final norm
        let h = self.norm.forward(&h);

        // Project to vocabulary: [batch, seq_len, vocab_size]
        self.lm_head.forward(&h)
    }

    /// Build a causal attention mask.
    ///
    /// Creates a lower-triangular mask where position i can attend to
    /// positions 0..=i but not future positions.
    fn build_causal_mask(seq_len: i32, offset: i32) -> Array {
        // Create a mask of shape [1, 1, seq_len, seq_len + offset]
        // where mask[..., i, j] = 0 if j <= i + offset, else -inf
        let total_len = seq_len + offset;

        // Row indices: [0, 1, ..., seq_len-1] + offset
        let rows = Array::arange(
            offset as f32,
            (offset + seq_len) as f32,
            1.0,
            crate::mlx::Dtype::Float32,
        );
        let rows = ops::reshape(&rows, &[seq_len, 1]);

        // Column indices: [0, 1, ..., total_len-1]
        let cols = Array::arange(0.0, total_len as f32, 1.0, crate::mlx::Dtype::Float32);
        let cols = ops::reshape(&cols, &[1, total_len]);

        // mask = where(cols <= rows, 0, -inf)
        let valid = ops::less(&cols, &ops::add(&rows, &Array::from_f32(1.0)));
        let neg_inf = Array::from_f32(f32::NEG_INFINITY);
        let zero = Array::from_f32(0.0);
        let mask = ops::where_cond(&valid, &zero, &neg_inf);

        // Expand to [1, 1, seq_len, total_len] for broadcasting
        ops::reshape(&mask, &[1, 1, seq_len, total_len])
    }

    /// Number of layers in the model.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}
