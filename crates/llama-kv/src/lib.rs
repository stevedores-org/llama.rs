//! # llama-kv
//!
//! First-class KV cache implementation for llama.rs.
//!
//! Supports:
//! - **Prefill**: Writes K/V tensors for a sequence of tokens `[seq_len, n_heads, head_dim]`
//! - **Decode**: Appends K/V for one token at a time, tracking cumulative sequence length
//! - **Multi-layer**: SessionKVCache manages K/V for all transformer layers with synchronized seq_len
//! - **Memory tracking**: Accurate byte accounting for both allocated and active memory
//! - **Type safety**: KVShape enforces valid dimensions

use std::fmt;

/// Represents tensor shape: `[seq_len, n_heads, head_dim]`.
///
/// Used for shape validation and memory calculations.
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

/// Layout strategy for KV cache memory organization.
///
/// Determines how K/V tensors are stored in memory:
/// - **BySequence**: Optimized for position-based access during decode
/// - **ByHead**: Optimized for per-head operations, aligns with Metal shared memory
/// - **Transposed**: Optimized for attention computation Q·K^T
///
/// Note: For Milestone A, all caches use BySequence layout. Layout flexibility
/// is preserved for future optimizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVLayout {
    BySequence,
    ByHead,
    Transposed,
}

/// Error type for KV cache operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KVError {
    #[error("KV cache capacity exceeded: {seq_len} > {max}")]
    CapacityExceeded { seq_len: usize, max: usize },

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("Cannot write prefill to non-empty cache (seq_len={seq_len}). Call clear() first")]
    NotEmpty { seq_len: usize },
}

pub type KVResult<T> = Result<T, KVError>;

/// Single transformer layer's K and V cache.
///
/// Stores K/V tensors with support for:
/// - Decode phase: `append_token()` adds one token at a time
/// - Prefill phase: `write_prefill()` writes all tokens at once
///
/// # Invariants
/// - All operations maintain `seq_len <= capacity`
/// - K and V tensors are always same size
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// K tensor, flattened: shape is `[seq_len, n_heads, head_dim]` -> `seq_len * n_heads * head_dim`
    pub k: Vec<f32>,
    /// V tensor, flattened: shape is `[seq_len, n_heads, head_dim]` -> `seq_len * n_heads * head_dim`
    pub v: Vec<f32>,

    /// Actual capacity (maximum seq_len this layer can hold)
    pub capacity: usize,
    /// Transformer layer configuration
    pub n_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Current sequence length (how many tokens are in the cache)
    pub seq_len: usize,

    /// Memory layout strategy (preserved for future use in optimization)
    pub layout: KVLayout,
}

impl LayerKVCache {
    /// Create a new KV cache for one transformer layer.
    ///
    /// # Arguments
    /// - `max_seq_len`: Maximum sequence length this cache can hold
    /// - `n_heads`: Number of attention heads
    /// - `head_dim`: Dimension of each head
    pub fn new(max_seq_len: usize, n_heads: usize, head_dim: usize) -> Self {
        let buf_len = max_seq_len * n_heads * head_dim;

        Self {
            k: vec![0.0; buf_len],
            v: vec![0.0; buf_len],
            capacity: max_seq_len,
            n_heads,
            head_dim,
            seq_len: 0,
            layout: KVLayout::BySequence,
        }
    }

    /// Append K and V for a single decode token.
    ///
    /// # Arguments
    /// - `k_token`: K tensor for this token, shape `[n_heads, head_dim]` (flattened, length = n_heads * head_dim)
    /// - `v_token`: V tensor for this token, shape `[n_heads, head_dim]` (flattened, length = n_heads * head_dim)
    ///
    /// # Errors
    /// - `CapacityExceeded`: If appending would exceed max sequence length
    /// - `ShapeMismatch`: If token shape doesn't match n_heads * head_dim
    pub fn append_token(&mut self, k_token: &[f32], v_token: &[f32]) -> KVResult<()> {
        let expected_len = self.n_heads * self.head_dim;

        if k_token.len() != expected_len || v_token.len() != expected_len {
            let got_len = if k_token.len() != expected_len {
                k_token.len()
            } else {
                v_token.len()
            };
            return Err(KVError::ShapeMismatch {
                expected: expected_len,
                got: got_len,
            });
        }

        if self.seq_len >= self.capacity {
            return Err(KVError::CapacityExceeded {
                seq_len: self.seq_len + 1,
                max: self.capacity,
            });
        }

        let offset = self.seq_len * self.n_heads * self.head_dim;
        self.k[offset..offset + expected_len].copy_from_slice(k_token);
        self.v[offset..offset + expected_len].copy_from_slice(v_token);
        self.seq_len += 1;

        Ok(())
    }

    /// Write K and V for an entire prefill sequence.
    ///
    /// # Arguments
    /// - `k_seq`: K tensor for entire sequence, shape `[prefill_len, n_heads, head_dim]` (flattened)
    /// - `v_seq`: V tensor for entire sequence, shape `[prefill_len, n_heads, head_dim]` (flattened)
    ///
    /// # Preconditions
    /// - Cache must be empty (`seq_len == 0`). For Milestone A, prefill always starts fresh.
    ///   To re-run prefill on an existing cache, call `clear()` first.
    ///
    /// # Errors
    /// - `NotEmpty`: If cache already contains tokens
    /// - `CapacityExceeded`: If prefill_len > max_seq_len
    /// - `ShapeMismatch`: If tensor shapes don't match expected dimensions
    pub fn write_prefill(&mut self, k_seq: &[f32], v_seq: &[f32]) -> KVResult<()> {
        if self.seq_len != 0 {
            return Err(KVError::NotEmpty {
                seq_len: self.seq_len,
            });
        }

        let prefill_len = k_seq.len() / (self.n_heads * self.head_dim);

        if prefill_len > self.capacity {
            return Err(KVError::CapacityExceeded {
                seq_len: prefill_len,
                max: self.capacity,
            });
        }

        let total_len = prefill_len * self.n_heads * self.head_dim;

        if k_seq.len() != total_len || v_seq.len() != total_len {
            let got_len = if k_seq.len() != total_len {
                k_seq.len()
            } else {
                v_seq.len()
            };
            return Err(KVError::ShapeMismatch {
                expected: total_len,
                got: got_len,
            });
        }

        self.k[..total_len].copy_from_slice(k_seq);
        self.v[..total_len].copy_from_slice(v_seq);
        self.seq_len = prefill_len;

        Ok(())
    }

    /// Clear the cache (reset seq_len to 0).
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Total memory allocated for K and V (in bytes).
    pub fn memory_bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f32>()
    }

    /// Memory used for currently written cache (based on seq_len, not capacity).
    ///
    /// Useful for tracking active memory vs. allocated capacity.
    /// Equal to `seq_len * n_heads * head_dim * 2 * sizeof(f32)` (K+V).
    pub fn active_memory_bytes(&self) -> usize {
        let used = self.seq_len * self.n_heads * self.head_dim;
        used * 2 * std::mem::size_of::<f32>()
    }
}

/// Session-level KV cache managing all transformer layers.
///
/// Ensures all layers maintain synchronized `seq_len` (invariant: all layers have same seq_len).
#[derive(Debug, Clone)]
pub struct SessionKVCache {
    layers: Vec<LayerKVCache>,
}

impl SessionKVCache {
    /// Create a new session KV cache for multiple layers.
    ///
    /// # Arguments
    /// - `n_layers`: Number of transformer layers (must be > 0)
    /// - `max_seq_len`: Maximum sequence length
    /// - `n_heads`: Number of attention heads (same for all layers)
    /// - `head_dim`: Head dimension (same for all layers)
    pub fn new(n_layers: usize, max_seq_len: usize, n_heads: usize, head_dim: usize) -> Self {
        assert!(n_layers > 0, "SessionKVCache requires n_layers > 0");
        let layers = (0..n_layers)
            .map(|_| LayerKVCache::new(max_seq_len, n_heads, head_dim))
            .collect();

        Self { layers }
    }

    /// Get KV cache for a specific layer (mutable).
    ///
    /// Crate-private to prevent callers from breaking the synchronized seq_len invariant.
    pub(crate) fn layer_mut(&mut self, layer_idx: usize) -> Option<&mut LayerKVCache> {
        self.layers.get_mut(layer_idx)
    }

    /// Get KV cache for a specific layer (immutable).
    pub fn layer(&self, layer_idx: usize) -> Option<&LayerKVCache> {
        self.layers.get(layer_idx)
    }

    /// Current sequence length. All layers must have the same seq_len (invariant).
    ///
    /// # Panics
    /// If SessionKVCache was created with `n_layers = 0`.
    pub fn seq_len(&self) -> usize {
        assert!(
            !self.layers.is_empty(),
            "SessionKVCache contains no layers; ensure it is constructed with n_layers > 0"
        );
        self.layers[0].seq_len
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Append K/V for one token to all layers.
    ///
    /// # Errors
    /// - Propagates errors from individual layer appends
    /// - After error, state is undefined; call `clear()` and retry from clean state
    pub fn append_token(&mut self, k_tokens: &[&[f32]], v_tokens: &[&[f32]]) -> KVResult<()> {
        if k_tokens.len() != self.layers.len() || v_tokens.len() != self.layers.len() {
            let got_len = if k_tokens.len() != self.layers.len() {
                k_tokens.len()
            } else {
                v_tokens.len()
            };
            return Err(KVError::ShapeMismatch {
                expected: self.layers.len(),
                got: got_len,
            });
        }

        // Pre-validate all layers before mutating any, to maintain atomic semantics.
        for (i, layer) in self.layers.iter().enumerate() {
            let expected_len = layer.n_heads * layer.head_dim;
            if k_tokens[i].len() != expected_len || v_tokens[i].len() != expected_len {
                let got_len = if k_tokens[i].len() != expected_len {
                    k_tokens[i].len()
                } else {
                    v_tokens[i].len()
                };
                return Err(KVError::ShapeMismatch {
                    expected: expected_len,
                    got: got_len,
                });
            }
            if layer.seq_len >= layer.capacity {
                return Err(KVError::CapacityExceeded {
                    seq_len: layer.seq_len + 1,
                    max: layer.capacity,
                });
            }
        }

        // All validations passed — perform writes.
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.append_token(k_tokens[i], v_tokens[i])?;
        }

        Ok(())
    }

    /// Write K/V for prefill across all layers.
    pub fn write_prefill(&mut self, k_seqs: &[&[f32]], v_seqs: &[&[f32]]) -> KVResult<()> {
        if k_seqs.len() != self.layers.len() || v_seqs.len() != self.layers.len() {
            let got_len = if k_seqs.len() != self.layers.len() {
                k_seqs.len()
            } else {
                v_seqs.len()
            };
            return Err(KVError::ShapeMismatch {
                expected: self.layers.len(),
                got: got_len,
            });
        }

        // Write layer-by-layer; on error, roll back to consistent empty state.
        for i in 0..self.layers.len() {
            if let Err(e) = self.layers[i].write_prefill(k_seqs[i], v_seqs[i]) {
                self.clear();
                return Err(e);
            }
        }

        Ok(())
    }

    /// Clear all layers.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Total memory across all layers.
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Active memory across all layers (based on seq_len).
    pub fn active_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.active_memory_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_shape_total_elements() {
        let shape = KVShape::new(128, 8, 64);
        assert_eq!(shape.total_elements(), 128 * 8 * 64);
    }

    #[test]
    fn layer_append_single_token() {
        let mut cache = LayerKVCache::new(10, 2, 4);
        let k_token = vec![0.1; 8];
        let v_token = vec![0.2; 8];

        assert_eq!(cache.seq_len, 0);
        cache.append_token(&k_token, &v_token).unwrap();
        assert_eq!(cache.seq_len, 1);
    }

    #[test]
    fn layer_append_multiple_tokens() {
        let mut cache = LayerKVCache::new(10, 2, 4);

        for i in 0..5 {
            let k_token = vec![0.1 * i as f32; 8];
            let v_token = vec![0.2 * i as f32; 8];
            cache.append_token(&k_token, &v_token).unwrap();
        }

        assert_eq!(cache.seq_len, 5);
    }

    #[test]
    fn layer_capacity_exceeded() {
        let mut cache = LayerKVCache::new(2, 2, 4);
        let k_token = vec![0.1; 8];
        let v_token = vec![0.2; 8];

        cache.append_token(&k_token, &v_token).unwrap();
        cache.append_token(&k_token, &v_token).unwrap();

        let result = cache.append_token(&k_token, &v_token);
        assert!(matches!(result, Err(KVError::CapacityExceeded { .. })));
    }

    #[test]
    fn layer_shape_mismatch_on_append() {
        let mut cache = LayerKVCache::new(10, 2, 4);
        let k_token = vec![0.1; 8];
        let v_token = vec![0.2; 7]; // Wrong length

        let result = cache.append_token(&k_token, &v_token);
        assert!(matches!(result, Err(KVError::ShapeMismatch { .. })));
    }

    #[test]
    fn layer_write_prefill() {
        let mut cache = LayerKVCache::new(10, 2, 4);
        let k_seq = vec![0.1; 32]; // 4 tokens * 2 heads * 4 dim
        let v_seq = vec![0.2; 32];

        cache.write_prefill(&k_seq, &v_seq).unwrap();
        assert_eq!(cache.seq_len, 4);
    }

    #[test]
    fn layer_prefill_not_empty_error() {
        let mut cache = LayerKVCache::new(10, 2, 4);
        let k_token = vec![0.1; 8];
        let v_token = vec![0.2; 8];
        let k_seq = vec![0.1; 16];
        let v_seq = vec![0.2; 16];

        // Append a token first
        cache.append_token(&k_token, &v_token).unwrap();

        // Try to write prefill (should fail)
        let result = cache.write_prefill(&k_seq, &v_seq);
        assert!(matches!(result, Err(KVError::NotEmpty { .. })));
    }

    #[test]
    fn layer_clear() {
        let mut cache = LayerKVCache::new(10, 2, 4);
        let k_token = vec![0.1; 8];
        let v_token = vec![0.2; 8];

        cache.append_token(&k_token, &v_token).unwrap();
        assert_eq!(cache.seq_len, 1);

        cache.clear();
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn layer_memory_bytes_calculation() {
        let cache = LayerKVCache::new(10, 2, 4);
        // K and V each have 10 * 2 * 4 = 80 f32s
        let expected_bytes = 2 * 80 * std::mem::size_of::<f32>();
        assert_eq!(cache.memory_bytes(), expected_bytes);
    }

    #[test]
    fn layer_active_memory_bytes() {
        let mut cache = LayerKVCache::new(10, 2, 4);
        let k_token = vec![0.1; 8];
        let v_token = vec![0.2; 8];

        // After appending 3 tokens
        cache.append_token(&k_token, &v_token).unwrap();
        cache.append_token(&k_token, &v_token).unwrap();
        cache.append_token(&k_token, &v_token).unwrap();

        // Active memory = 3 tokens * 2 heads * 4 dim * 2 (K+V) * sizeof(f32)
        let expected = 3 * 2 * 4 * 2 * std::mem::size_of::<f32>();
        assert_eq!(cache.active_memory_bytes(), expected);
    }

    #[test]
    fn session_seq_len_synchronized() {
        let mut session = SessionKVCache::new(3, 10, 2, 4);
        assert_eq!(session.seq_len(), 0);

        let k_tokens = vec![vec![0.1; 8], vec![0.1; 8], vec![0.1; 8]];
        let v_tokens = vec![vec![0.2; 8], vec![0.2; 8], vec![0.2; 8]];

        session
            .append_token(
                &k_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                &v_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            )
            .unwrap();

        // All layers should have seq_len = 1
        for i in 0..3 {
            assert_eq!(session.layer(i).unwrap().seq_len, 1);
        }
        assert_eq!(session.seq_len(), 1);
    }

    #[test]
    fn session_n_layers() {
        let session = SessionKVCache::new(5, 10, 2, 4);
        assert_eq!(session.n_layers(), 5);
    }

    #[test]
    fn session_clear() {
        let mut session = SessionKVCache::new(2, 10, 2, 4);
        let k_tokens = vec![vec![0.1; 8], vec![0.1; 8]];
        let v_tokens = vec![vec![0.2; 8], vec![0.2; 8]];

        session
            .append_token(
                &k_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                &v_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            )
            .unwrap();

        assert_eq!(session.seq_len(), 1);
        session.clear();
        assert_eq!(session.seq_len(), 0);
    }

    #[test]
    fn session_memory_bytes() {
        let session = SessionKVCache::new(2, 10, 2, 4);
        let single_layer_memory = 2 * 10 * 2 * 4 * std::mem::size_of::<f32>();
        let expected = single_layer_memory * 2;
        assert_eq!(session.memory_bytes(), expected);
    }

    #[test]
    fn session_active_memory_bytes() {
        let mut session = SessionKVCache::new(2, 10, 2, 4);
        let k_tokens = vec![vec![0.1; 8], vec![0.1; 8]];
        let v_tokens = vec![vec![0.2; 8], vec![0.2; 8]];

        for _ in 0..2 {
            session
                .append_token(
                    &k_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                    &v_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                )
                .unwrap();
        }

        // 2 layers * 2 tokens * 2 heads * 4 dim * 2 (K+V) * sizeof(f32)
        let expected = 2 * 2 * 2 * 4 * 2 * std::mem::size_of::<f32>();
        assert_eq!(session.active_memory_bytes(), expected);
    }

    #[test]
    fn session_append_token_atomic_on_capacity_error() {
        // 2 layers, capacity=1: fill first, then attempt second append
        let mut session = SessionKVCache::new(2, 1, 2, 4);
        let k_tokens: Vec<Vec<f32>> = vec![vec![0.1; 8], vec![0.1; 8]];
        let v_tokens: Vec<Vec<f32>> = vec![vec![0.2; 8], vec![0.2; 8]];

        session
            .append_token(
                &k_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                &v_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            )
            .unwrap();
        assert_eq!(session.seq_len(), 1);

        // This should fail atomically — no layer should advance.
        let result = session.append_token(
            &k_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            &v_tokens.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
        );
        assert!(result.is_err());

        // Both layers should still be at seq_len=1 (unchanged).
        for i in 0..2 {
            assert_eq!(session.layer(i).unwrap().seq_len, 1);
        }
    }

    #[test]
    fn session_write_prefill_rollback_on_error() {
        // 2 layers, capacity=4. Give layer 0 valid data, layer 1 wrong-shape data.
        let mut session = SessionKVCache::new(2, 4, 2, 4);
        let k_good = vec![0.1; 32]; // 4 tokens * 2 heads * 4 dim
        let v_good = vec![0.2; 32];
        let k_bad = vec![0.1; 17]; // Wrong shape
        let v_bad = vec![0.2; 17];

        let result = session.write_prefill(
            &[k_good.as_slice(), k_bad.as_slice()],
            &[v_good.as_slice(), v_bad.as_slice()],
        );
        assert!(result.is_err());

        // After rollback, all layers should be at seq_len=0.
        for i in 0..2 {
            assert_eq!(session.layer(i).unwrap().seq_len, 0);
        }
    }

    #[test]
    #[should_panic(expected = "n_layers > 0")]
    fn session_zero_layers_panics() {
        SessionKVCache::new(0, 10, 2, 4);
    }

    #[test]
    fn session_prefill_across_layers() {
        let mut session = SessionKVCache::new(2, 10, 2, 4);
        let k_seq = vec![0.1; 32]; // 4 tokens
        let v_seq = vec![0.2; 32];

        session
            .write_prefill(
                &[k_seq.as_slice(), k_seq.as_slice()],
                &[v_seq.as_slice(), v_seq.as_slice()],
            )
            .unwrap();

        assert_eq!(session.seq_len(), 4);
    }
}
