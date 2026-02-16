//! # llama-kv
//!
//! First-class KV cache implementation for llama.rs.
//!
//! Supports:
//! - Prefill: writes K/V for `[seq_len, n_heads, head_dim]`
//! - Decode: appends 1 token at a time
//! - Memory-friendly layouts for Metal/CPU
//! - Future: paging/eviction, sliding window

use std::fmt;

/// Represents a shape: `[seq_len, heads, head_dim]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KVShape {
    pub seq_len: usize,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl KVShape {
    pub fn new(seq_len: usize, n_heads: usize, head_dim: usize) -> Self {
        Self {
            seq_len,
            n_heads,
            head_dim,
        }
    }

    pub fn total_elements(&self) -> usize {
        self.seq_len * self.n_heads * self.head_dim
    }

    pub fn capacity_bytes(&self, bytes_per_element: usize) -> usize {
        self.total_elements() * bytes_per_element
    }
}

impl fmt::Display for KVShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[seq:{}, heads:{}, dim:{}]",
            self.seq_len, self.n_heads, self.head_dim
        )
    }
}

/// Layout policy for KV cache memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVLayout {
    /// Contiguous by sequence position: `[seq_len][heads][head_dim]`
    /// Good for positional access during decode.
    BySequence,

    /// Contiguous by head: `[heads][seq_len][head_dim]`
    /// Good for per-head operations, Metal alignment.
    ByHead,

    /// Transposed for attention: `[heads][head_dim][seq_len]`
    /// Optimizes attention Q·K^T computation.
    Transposed,
}

/// Reference KV cache. Stores K and V tensors for a single layer.
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// K tensor: shape [seq_len, n_heads, head_dim], typically f32
    pub k: Vec<f32>,

    /// V tensor: shape [seq_len, n_heads, head_dim], typically f32
    pub v: Vec<f32>,

    /// Current sequence length written to cache.
    pub seq_len: usize,

    /// Maximum capacity before reallocation.
    pub max_seq_len: usize,

    /// Configuration.
    pub n_heads: usize,
    pub head_dim: usize,

    /// Layout policy.
    pub layout: KVLayout,
}

impl LayerKVCache {
    /// Create a new KV cache for a single layer.
    pub fn new(max_seq_len: usize, n_heads: usize, head_dim: usize, layout: KVLayout) -> Self {
        let total = max_seq_len * n_heads * head_dim;

        Self {
            k: vec![0.0; total],
            v: vec![0.0; total],
            seq_len: 0,
            max_seq_len,
            n_heads,
            head_dim,
            layout,
        }
    }

    /// Get shape of current cache state.
    pub fn shape(&self) -> KVShape {
        KVShape::new(self.seq_len, self.n_heads, self.head_dim)
    }

    /// Append K and V for a single token. Used during decode.
    ///
    /// Expects `k_token` and `v_token` to be `[n_heads, head_dim]`.
    /// This appends to position `self.seq_len` and increments it.
    pub fn append_token(&mut self, k_token: &[f32], v_token: &[f32]) -> Result<(), KVError> {
        if self.seq_len >= self.max_seq_len {
            return Err(KVError::CapacityExceeded {
                seq_len: self.seq_len,
                max: self.max_seq_len,
            });
        }

        if k_token.len() != self.n_heads * self.head_dim {
            return Err(KVError::ShapeMismatch {
                expected: self.n_heads * self.head_dim,
                got: k_token.len(),
            });
        }

        if v_token.len() != self.n_heads * self.head_dim {
            return Err(KVError::ShapeMismatch {
                expected: self.n_heads * self.head_dim,
                got: v_token.len(),
            });
        }

        let offset = self.seq_len * self.n_heads * self.head_dim;

        // Copy into cache at current position.
        self.k[offset..offset + k_token.len()].copy_from_slice(k_token);
        self.v[offset..offset + v_token.len()].copy_from_slice(v_token);

        self.seq_len += 1;

        Ok(())
    }

    /// Write K and V for a sequence of positions. Used during prefill.
    ///
    /// Expects `k_seq` to be `[seq_len, n_heads, head_dim]`.
    pub fn write_prefill(
        &mut self,
        k_seq: &[f32],
        v_seq: &[f32],
        prefill_len: usize,
    ) -> Result<(), KVError> {
        if self.seq_len != 0 {
            return Err(KVError::NotEmpty);
        }

        if prefill_len > self.max_seq_len {
            return Err(KVError::CapacityExceeded {
                seq_len: prefill_len,
                max: self.max_seq_len,
            });
        }

        let expected_len = prefill_len * self.n_heads * self.head_dim;

        if k_seq.len() != expected_len || v_seq.len() != expected_len {
            return Err(KVError::ShapeMismatch {
                expected: expected_len,
                got: k_seq.len(),
            });
        }

        self.k[..expected_len].copy_from_slice(k_seq);
        self.v[..expected_len].copy_from_slice(v_seq);
        self.seq_len = prefill_len;

        Ok(())
    }

    /// Clear the cache (reset seq_len to 0).
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Memory used (in bytes, assuming f32 = 4 bytes).
    pub fn memory_bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 4
    }

    /// Memory used for currently written cache (seq_len, not capacity).
    pub fn memory_used_bytes(&self) -> usize {
        let used = self.seq_len * self.n_heads * self.head_dim;
        used * 8 // 4 bytes for K + 4 bytes for V
    }
}

/// Error type for KV operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KVError {
    CapacityExceeded { seq_len: usize, max: usize },
    ShapeMismatch { expected: usize, got: usize },
    NotEmpty,
}

impl fmt::Display for KVError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KVError::CapacityExceeded { seq_len, max } => {
                write!(f, "KV cache capacity exceeded: {} > {}", seq_len, max)
            }
            KVError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {}, got {}", expected, got)
            }
            KVError::NotEmpty => write!(f, "KV cache must be empty for prefill"),
        }
    }
}

impl std::error::Error for KVError {}

/// A session-level KV cache for all layers.
#[derive(Debug, Clone)]
pub struct SessionKVCache {
    /// One cache per transformer layer.
    pub layers: Vec<LayerKVCache>,
}

impl SessionKVCache {
    /// Create a full KV cache for a model with `n_layers` transformer blocks.
    pub fn new(
        n_layers: usize,
        max_seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        layout: KVLayout,
    ) -> Self {
        let layers = (0..n_layers)
            .map(|_| LayerKVCache::new(max_seq_len, n_heads, head_dim, layout))
            .collect();

        Self { layers }
    }

    /// Get current sequence length (should be same for all layers).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
    }

    /// Clear all layers.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Total memory for all layers.
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Memory actively used by current seq_len.
    pub fn memory_used_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_used_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_shape_calculations() {
        let shape = KVShape::new(128, 32, 128);
        assert_eq!(shape.total_elements(), 128 * 32 * 128);
        assert_eq!(shape.capacity_bytes(4), 128 * 32 * 128 * 4);
    }

    #[test]
    fn layer_kv_new() {
        let cache = LayerKVCache::new(256, 32, 128, KVLayout::BySequence);
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.shape().seq_len, 0);
    }

    #[test]
    fn layer_kv_append_token() {
        let mut cache = LayerKVCache::new(256, 2, 4, KVLayout::BySequence);
        let k_token = vec![1.0; 8]; // 2 heads * 4 dim
        let v_token = vec![2.0; 8];

        cache.append_token(&k_token, &v_token).unwrap();
        assert_eq!(cache.seq_len, 1);

        cache.append_token(&k_token, &v_token).unwrap();
        assert_eq!(cache.seq_len, 2);
    }

    #[test]
    fn layer_kv_capacity_exceeded() {
        let mut cache = LayerKVCache::new(2, 2, 4, KVLayout::BySequence);
        let k_token = vec![1.0; 8];
        let v_token = vec![2.0; 8];

        cache.append_token(&k_token, &v_token).unwrap();
        cache.append_token(&k_token, &v_token).unwrap();

        let err = cache.append_token(&k_token, &v_token);
        assert!(matches!(
            err,
            Err(KVError::CapacityExceeded { seq_len: 2, max: 2 })
        ));
    }

    #[test]
    fn layer_kv_shape_mismatch() {
        let mut cache = LayerKVCache::new(256, 2, 4, KVLayout::BySequence);
        let k_token = vec![1.0; 7]; // Wrong size
        let v_token = vec![2.0; 8];

        let err = cache.append_token(&k_token, &v_token);
        assert!(matches!(err, Err(KVError::ShapeMismatch { .. })));
    }

    #[test]
    fn layer_kv_write_prefill() {
        let mut cache = LayerKVCache::new(256, 2, 4, KVLayout::BySequence);

        let k_seq = vec![1.0; 4 * 2 * 4]; // seq_len=4, heads=2, dim=4
        let v_seq = vec![2.0; 4 * 2 * 4];

        cache.write_prefill(&k_seq, &v_seq, 4).unwrap();
        assert_eq!(cache.seq_len, 4);
    }

    #[test]
    fn layer_kv_clear() {
        let mut cache = LayerKVCache::new(256, 2, 4, KVLayout::BySequence);
        let k_token = vec![1.0; 8];
        let v_token = vec![2.0; 8];

        cache.append_token(&k_token, &v_token).unwrap();
        assert_eq!(cache.seq_len, 1);

        cache.clear();
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn session_kv_cache_all_layers() {
        let mut session = SessionKVCache::new(8, 256, 32, 128, KVLayout::BySequence);
        assert_eq!(session.layers.len(), 8);

        let k_token = vec![1.0; 32 * 128];
        let v_token = vec![2.0; 32 * 128];

        for layer in &mut session.layers {
            layer.append_token(&k_token, &v_token).unwrap();
        }

        assert_eq!(session.seq_len(), 1);
    }

    #[test]
    fn memory_calculations() {
        let cache = LayerKVCache::new(256, 32, 128, KVLayout::BySequence);
        let bytes = cache.memory_bytes();
        assert_eq!(bytes, 256 * 32 * 128 * 2 * 4); // 2 tensors, 4 bytes each
    }
}
