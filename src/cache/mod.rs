//! Key-Value cache for efficient autoregressive inference.
//!
//! The KV cache stores computed Key and Value vectors from previous tokens,
//! avoiding redundant computation during the decode phase.
//!
//! # Strategy: Pre-allocated Cache (Recommended)
//!
//! Instead of concatenating new KV pairs each step (which requires O(N) copy),
//! we pre-allocate a fixed-size buffer and use slice_update for O(1) updates.
//! This eliminates allocation during the generation loop and prevents Metal
//! kernel recompilation due to changing tensor shapes.
//!
//! # Memory Layout
//!
//! Keys:   `[batch, max_seq_len, n_kv_heads, head_dim]`
//! Values: `[batch, max_seq_len, n_kv_heads, head_dim]`

use crate::mlx::ops;
use crate::mlx::{Array, Dtype};

/// Pre-allocated KV cache for a single transformer layer.
pub struct KvCache {
    /// Cached key states, shape `[batch, max_seq_len, n_kv_heads, head_dim]`.
    keys: Array,

    /// Cached value states, shape `[batch, max_seq_len, n_kv_heads, head_dim]`.
    values: Array,

    /// Current write position (number of tokens cached so far).
    offset: usize,

    /// Maximum sequence length this cache can hold.
    max_seq_len: usize,

    /// Batch size.
    batch_size: usize,

    /// Number of KV heads.
    n_kv_heads: usize,

    /// Dimension of each head.
    head_dim: usize,
}

impl KvCache {
    /// Create a new pre-allocated KV cache.
    ///
    /// # Arguments
    /// - `batch_size`: Batch size (typically 1 for interactive inference).
    /// - `max_seq_len`: Maximum context length to support.
    /// - `n_kv_heads`: Number of key-value heads.
    /// - `head_dim`: Dimension of each attention head.
    /// - `dtype`: Data type for cache storage (should match model precision).
    pub fn new(
        batch_size: usize,
        max_seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: Dtype,
    ) -> Self {
        let shape = [
            batch_size as i32,
            max_seq_len as i32,
            n_kv_heads as i32,
            head_dim as i32,
        ];

        let keys = Array::zeros(&shape, dtype);
        let values = Array::zeros(&shape, dtype);

        KvCache {
            keys,
            values,
            offset: 0,
            max_seq_len,
            batch_size,
            n_kv_heads,
            head_dim,
        }
    }

    /// Update the cache with new key and value tensors.
    ///
    /// Uses `slice_update` for O(1) cache updates without allocation.
    ///
    /// # Arguments
    /// - `new_keys`: New key states `[batch, seq_len, n_kv_heads, head_dim]`.
    /// - `new_values`: New value states `[batch, seq_len, n_kv_heads, head_dim]`.
    pub fn update(&mut self, new_keys: &Array, new_values: &Array) {
        let new_seq_len = new_keys.shape()[1] as usize;

        // Slice update: write into [batch, offset:offset+seq_len, heads, dim]
        let start = [0, self.offset as i32, 0, 0];
        let stop = [
            self.batch_size as i32,
            (self.offset + new_seq_len) as i32,
            self.n_kv_heads as i32,
            self.head_dim as i32,
        ];
        let strides = [1, 1, 1, 1];

        self.keys = ops::slice_update(&self.keys, new_keys, &start, &stop, &strides);
        self.values = ops::slice_update(&self.values, new_values, &start, &stop, &strides);

        self.offset += new_seq_len;
    }

    /// Get the cached keys and values up to the current offset.
    ///
    /// Returns slices `[batch, 0:offset, n_kv_heads, head_dim]`.
    pub fn get(&self) -> (Array, Array) {
        let start = [0, 0, 0, 0];
        let stop = [
            self.batch_size as i32,
            self.offset as i32,
            self.n_kv_heads as i32,
            self.head_dim as i32,
        ];
        let strides = [1, 1, 1, 1];

        let k = ops::slice(&self.keys, &start, &stop, &strides);
        let v = ops::slice(&self.values, &start, &stop, &strides);
        (k, v)
    }

    /// Current number of cached tokens.
    pub fn len(&self) -> usize {
        self.offset
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Maximum capacity of the cache.
    pub fn capacity(&self) -> usize {
        self.max_seq_len
    }

    /// Remaining capacity.
    pub fn remaining(&self) -> usize {
        self.max_seq_len - self.offset
    }

    /// Reset the cache (for a new conversation).
    ///
    /// Re-allocates zero tensors and resets the write offset.
    pub fn reset(&mut self) {
        let shape = [
            self.batch_size as i32,
            self.max_seq_len as i32,
            self.n_kv_heads as i32,
            self.head_dim as i32,
        ];
        // Determine dtype from existing keys
        let dtype = self.keys.dtype().unwrap_or(Dtype::Float16);
        self.keys = Array::zeros(&shape, dtype);
        self.values = Array::zeros(&shape, dtype);
        self.offset = 0;
    }

    /// Estimated memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let dtype = self.keys.dtype().unwrap_or(Dtype::Float16);
        let elements = self.batch_size * self.max_seq_len * self.n_kv_heads * self.head_dim;
        // keys + values
        elements * 2 * dtype.size_bytes()
    }
}

/// Create KV caches for all layers in the model.
pub fn create_caches(
    num_layers: usize,
    batch_size: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    dtype: Dtype,
) -> Vec<KvCache> {
    (0..num_layers)
        .map(|_| KvCache::new(batch_size, max_seq_len, n_kv_heads, head_dim, dtype))
        .collect()
}

/// Reset all caches (for a new conversation).
pub fn reset_all_caches(caches: &mut [KvCache]) {
    for cache in caches.iter_mut() {
        cache.reset();
    }
}
