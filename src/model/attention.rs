//! Grouped Query Attention (GQA) for Llama 3.
//!
//! GQA reduces the KV cache memory footprint by using fewer KV heads than
//! query heads. For example, Llama 3 8B uses 32 query heads but only 8 KV
//! heads (4:1 ratio).
//!
//! The attention computation uses MLX's `scaled_dot_product_attention` which
//! maps to the FlashAttention Metal kernel, avoiding materialization of the
//! full N x N attention matrix.

use crate::cache::KvCache;
use crate::mlx::ops;
use crate::mlx::Array;
use crate::model::config::ModelConfig;
use crate::model::linear::AnyLinear;
use crate::model::rope::{self, RopeConfig};

/// Grouped Query Attention layer.
pub struct Attention {
    /// Query projection: [hidden_size] -> [n_heads * head_dim].
    pub q_proj: AnyLinear,

    /// Key projection: [hidden_size] -> [n_kv_heads * head_dim].
    pub k_proj: AnyLinear,

    /// Value projection: [hidden_size] -> [n_kv_heads * head_dim].
    pub v_proj: AnyLinear,

    /// Output projection: [n_heads * head_dim] -> [hidden_size].
    pub o_proj: AnyLinear,

    /// RoPE configuration.
    pub rope_config: RopeConfig,

    /// Number of query heads.
    pub n_heads: usize,

    /// Number of key-value heads.
    pub n_kv_heads: usize,

    /// Dimension of each head.
    pub head_dim: usize,

    /// Attention scale factor: 1 / sqrt(head_dim).
    pub scale: f32,
}

impl Attention {
    pub fn new(
        q_proj: AnyLinear,
        k_proj: AnyLinear,
        v_proj: AnyLinear,
        o_proj: AnyLinear,
        config: &ModelConfig,
    ) -> Self {
        let head_dim = config.head_dim;
        Attention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope_config: RopeConfig {
                dims: head_dim as i32,
                traditional: config.rope_traditional,
                base: config.rope_theta,
                scale: config.rope_scaling,
            },
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Forward pass for the attention layer.
    ///
    /// # Arguments
    /// - `x`: Input tensor `[batch, seq_len, hidden_size]`.
    /// - `cache`: Mutable reference to the KV cache for this layer.
    /// - `offset`: Current position offset for RoPE.
    /// - `mask`: Optional causal attention mask.
    ///
    /// # Returns
    /// Output tensor `[batch, seq_len, hidden_size]`.
    pub fn forward(
        &self,
        x: &Array,
        cache: &mut KvCache,
        offset: i32,
        mask: Option<&Array>,
    ) -> Array {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project to Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape to multi-head format: [batch, seq_len, n_heads, head_dim]
        let q = ops::reshape(&q, &[
            batch,
            seq_len,
            self.n_heads as i32,
            self.head_dim as i32,
        ]);
        let k = ops::reshape(&k, &[
            batch,
            seq_len,
            self.n_kv_heads as i32,
            self.head_dim as i32,
        ]);
        let v = ops::reshape(&v, &[
            batch,
            seq_len,
            self.n_kv_heads as i32,
            self.head_dim as i32,
        ]);

        // Apply RoPE to queries and keys.
        // offset is 0 during prefill, seq_position during decode.
        let q = rope::apply_rope(&q, &self.rope_config, offset);
        let k = rope::apply_rope(&k, &self.rope_config, offset);

        // Update KV cache: write new K, V at the current offset.
        cache.update(&k, &v);

        // Read the full cached K, V (all positions up to current).
        let (cached_k, cached_v) = cache.get();

        // Transpose to [batch, n_heads, seq_len, head_dim] for attention.
        let q = ops::transpose(&q, &[0, 2, 1, 3]);
        let cached_k = ops::transpose(&cached_k, &[0, 2, 1, 3]);
        let cached_v = ops::transpose(&cached_v, &[0, 2, 1, 3]);

        // If using GQA, repeat KV heads to match query heads.
        let (cached_k, cached_v) = if self.n_kv_heads < self.n_heads {
            let repeats = (self.n_heads / self.n_kv_heads) as i32;
            (
                ops::repeat(&cached_k, repeats, 1),
                ops::repeat(&cached_v, repeats, 1),
            )
        } else {
            (cached_k, cached_v)
        };

        // Scaled dot-product attention (FlashAttention on Metal).
        let attn_out = ops::scaled_dot_product_attention(
            &q,
            &cached_k,
            &cached_v,
            self.scale,
            mask,
        );

        // Transpose back: [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, n_heads, head_dim]
        let attn_out = ops::transpose(&attn_out, &[0, 2, 1, 3]);

        // Merge heads: [batch, seq_len, n_heads * head_dim]
        let attn_out = ops::reshape(&attn_out, &[
            batch,
            seq_len,
            (self.n_heads * self.head_dim) as i32,
        ]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }
}
