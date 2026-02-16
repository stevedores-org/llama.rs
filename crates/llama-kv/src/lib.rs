//! # llama-kv
//!
//! First-class KV cache implementation for llama.rs.
//!
//! Supports:
//! - Prefill: writes K/V for `[seq_len, n_heads, head_dim]`
//! - Decode: appends 1 token at a time
//! - Memory-friendly layouts for Metal/CPU
//! - Sliding-window eviction via `KvPager` trait

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

impl std::fmt::Display for KVShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

/// Paging/eviction interface for sliding-window KV behavior.
pub trait KvPager {
    /// Keep at most `window` most-recent tokens, evicting older entries.
    /// Returns the number of evicted tokens.
    fn apply_sliding_window(&mut self, window: usize) -> Result<usize, KVError>;
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

    fn index(&self, seq: usize, head: usize, dim: usize) -> usize {
        match self.layout {
            KVLayout::BySequence => (seq * self.n_heads + head) * self.head_dim + dim,
            KVLayout::ByHead => (head * self.max_seq_len + seq) * self.head_dim + dim,
            KVLayout::Transposed => (head * self.head_dim + dim) * self.max_seq_len + seq,
        }
    }

    fn write_token_at_position(&mut self, seq: usize, k_token: &[f32], v_token: &[f32]) {
        for head in 0..self.n_heads {
            for dim in 0..self.head_dim {
                let src = head * self.head_dim + dim;
                let dst = self.index(seq, head, dim);
                self.k[dst] = k_token[src];
                self.v[dst] = v_token[src];
            }
        }
    }

    fn read_token_at_position(&self, seq: usize, k_out: &mut [f32], v_out: &mut [f32]) {
        for head in 0..self.n_heads {
            for dim in 0..self.head_dim {
                let dst = head * self.head_dim + dim;
                let src = self.index(seq, head, dim);
                k_out[dst] = self.k[src];
                v_out[dst] = self.v[src];
            }
        }
    }

    /// Append K and V for a single token. Used during decode.
    ///
    /// Expects `k_token` and `v_token` to be `[n_heads, head_dim]`.
    /// This appends to position `self.seq_len` and increments it.
    pub fn append_token(&mut self, k_token: &[f32], v_token: &[f32]) -> Result<(), KVError> {
        if self.seq_len >= self.max_seq_len {
            return Err(KVError::CapacityExceeded {
                seq_len: self.seq_len + 1,
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

        self.write_token_at_position(self.seq_len, k_token, v_token);

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
            let got_len = if k_seq.len() != expected_len {
                k_seq.len()
            } else {
                v_seq.len()
            };
            return Err(KVError::ShapeMismatch {
                expected: expected_len,
                got: got_len,
            });
        }

        for seq in 0..prefill_len {
            let src_offset = seq * self.n_heads * self.head_dim;
            let src_end = src_offset + self.n_heads * self.head_dim;
            self.write_token_at_position(
                seq,
                &k_seq[src_offset..src_end],
                &v_seq[src_offset..src_end],
            );
        }
        self.seq_len = prefill_len;

        Ok(())
    }

    /// Clear the cache (reset seq_len to 0).
    ///
    /// This does not zeroize `k`/`v`; old values remain in backing storage and
    /// will be overwritten by future writes.
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Evict the oldest `count` tokens from cache, compacting remaining tokens to
    /// the front. This defines the sliding-window behavior for all layouts.
    pub fn evict_prefix(&mut self, count: usize) -> Result<(), KVError> {
        if count > self.seq_len {
            return Err(KVError::EvictionOutOfRange {
                requested: count,
                available: self.seq_len,
            });
        }
        if count == 0 {
            return Ok(());
        }
        if count == self.seq_len {
            self.clear();
            return Ok(());
        }

        let keep = self.seq_len - count;
        let token_width = self.n_heads * self.head_dim;
        let mut k_tmp = vec![0.0f32; token_width];
        let mut v_tmp = vec![0.0f32; token_width];

        for new_seq in 0..keep {
            let old_seq = new_seq + count;
            self.read_token_at_position(old_seq, &mut k_tmp, &mut v_tmp);
            self.write_token_at_position(new_seq, &k_tmp, &v_tmp);
        }

        self.seq_len = keep;
        Ok(())
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
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KVError {
    #[error("KV cache capacity exceeded: {seq_len} > {max}")]
    CapacityExceeded { seq_len: usize, max: usize },
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },
    #[error("KV cache must be empty for prefill")]
    NotEmpty,
    #[error("eviction out of range: requested {requested}, available {available}")]
    EvictionOutOfRange { requested: usize, available: usize },
}

impl KvPager for LayerKVCache {
    fn apply_sliding_window(&mut self, window: usize) -> Result<usize, KVError> {
        if self.seq_len <= window {
            return Ok(0);
        }
        let evict = self.seq_len - window;
        self.evict_prefix(evict)?;
        Ok(evict)
    }
}

/// A session-level KV cache for all layers.
///
/// Maintains a synchronized `seq_len` invariant: all layers have the same seq_len.
#[derive(Debug, Clone)]
pub struct SessionKVCache {
    /// One cache per transformer layer.
    layers: Vec<LayerKVCache>,
}

impl SessionKVCache {
    /// Create a full KV cache for a model with `n_layers` transformer blocks.
    ///
    /// # Panics
    /// If `n_layers` is 0.
    pub fn new(
        n_layers: usize,
        max_seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        layout: KVLayout,
    ) -> Self {
        assert!(n_layers > 0, "SessionKVCache requires n_layers > 0");
        let layers = (0..n_layers)
            .map(|_| LayerKVCache::new(max_seq_len, n_heads, head_dim, layout))
            .collect();

        Self { layers }
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get immutable reference to a layer's cache.
    pub fn layer(&self, idx: usize) -> Option<&LayerKVCache> {
        self.layers.get(idx)
    }

    /// Get mutable reference to a layer's cache (crate-private to protect invariants).
    pub(crate) fn layer_mut(&mut self, idx: usize) -> Option<&mut LayerKVCache> {
        self.layers.get_mut(idx)
    }

    /// Get current sequence length (same for all layers by invariant).
    pub fn seq_len(&self) -> usize {
        self.layers[0].seq_len
    }

    /// Append K/V for one token to all layers atomically.
    ///
    /// Pre-validates all layers before writing to maintain synchronized seq_len.
    pub fn append_token(
        &mut self,
        k_tokens: &[&[f32]],
        v_tokens: &[&[f32]],
    ) -> Result<(), KVError> {
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

        // Pre-validate all layers before mutating any.
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.seq_len >= layer.max_seq_len {
                return Err(KVError::CapacityExceeded {
                    seq_len: layer.seq_len + 1,
                    max: layer.max_seq_len,
                });
            }
            let expected = layer.n_heads * layer.head_dim;
            if k_tokens[i].len() != expected {
                return Err(KVError::ShapeMismatch {
                    expected,
                    got: k_tokens[i].len(),
                });
            }
            if v_tokens[i].len() != expected {
                return Err(KVError::ShapeMismatch {
                    expected,
                    got: v_tokens[i].len(),
                });
            }
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.append_token(k_tokens[i], v_tokens[i])?;
        }

        Ok(())
    }

    /// Write prefill K/V across all layers with rollback on error.
    pub fn write_prefill(
        &mut self,
        k_seqs: &[&[f32]],
        v_seqs: &[&[f32]],
        prefill_len: usize,
    ) -> Result<(), KVError> {
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

        for i in 0..self.layers.len() {
            let result =
                self.layer_mut(i)
                    .unwrap()
                    .write_prefill(k_seqs[i], v_seqs[i], prefill_len);
            if let Err(e) = result {
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

    /// Total memory for all layers.
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Memory actively used by current seq_len.
    pub fn memory_used_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_used_bytes()).sum()
    }
}

impl KvPager for SessionKVCache {
    fn apply_sliding_window(&mut self, window: usize) -> Result<usize, KVError> {
        let evict = self
            .layers
            .first()
            .map(|l| l.seq_len.saturating_sub(window))
            .unwrap_or(0);
        // Validate all layers before mutating any to avoid partial eviction.
        for layer in &self.layers {
            if evict > layer.seq_len {
                return Err(KVError::EvictionOutOfRange {
                    requested: evict,
                    available: layer.seq_len,
                });
            }
        }
        for layer in &mut self.layers {
            layer.evict_prefix(evict)?;
        }
        Ok(evict)
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
            Err(KVError::CapacityExceeded { seq_len: 3, max: 2 })
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
    fn layer_kv_shape_mismatch_reports_v_length() {
        let mut cache = LayerKVCache::new(256, 2, 4, KVLayout::BySequence);
        let k_token = vec![1.0; 8]; // Correct
        let v_token = vec![2.0; 7]; // Wrong size

        let err = cache.append_token(&k_token, &v_token);
        assert!(matches!(
            err,
            Err(KVError::ShapeMismatch {
                expected: 8,
                got: 7
            })
        ));
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
        assert_eq!(session.n_layers(), 8);

        let k_token = vec![1.0; 32 * 128];
        let v_token = vec![2.0; 32 * 128];

        let k_tokens: Vec<&[f32]> = (0..8).map(|_| k_token.as_slice()).collect();
        let v_tokens: Vec<&[f32]> = (0..8).map(|_| v_token.as_slice()).collect();

        session.append_token(&k_tokens, &v_tokens).unwrap();

        assert_eq!(session.seq_len(), 1);
    }

    #[test]
    fn memory_calculations() {
        let cache = LayerKVCache::new(256, 32, 128, KVLayout::BySequence);
        let bytes = cache.memory_bytes();
        assert_eq!(bytes, 256 * 32 * 128 * 2 * 4); // 2 tensors, 4 bytes each
    }

    #[test]
    fn layout_by_head_changes_memory_indexing() {
        let mut cache = LayerKVCache::new(4, 2, 2, KVLayout::ByHead);
        cache
            .append_token(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .unwrap();

        // head 0, seq 0
        assert_eq!(cache.k[0], 1.0);
        assert_eq!(cache.k[1], 2.0);
        // head 1, seq 0 starts at head_stride = max_seq_len * head_dim = 8
        assert_eq!(cache.k[8], 3.0);
        assert_eq!(cache.k[9], 4.0);
    }

    #[test]
    fn layout_transposed_changes_memory_indexing() {
        let mut cache = LayerKVCache::new(4, 2, 2, KVLayout::Transposed);
        cache
            .append_token(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .unwrap();

        // h0,d0,s0 and h0,d1,s0
        assert_eq!(cache.k[0], 1.0);
        assert_eq!(cache.k[4], 2.0);
        // h1,d0,s0 and h1,d1,s0
        assert_eq!(cache.k[8], 3.0);
        assert_eq!(cache.k[12], 4.0);
    }

    #[test]
    fn evict_prefix_compacts_oldest_tokens() {
        let mut cache = LayerKVCache::new(8, 1, 2, KVLayout::BySequence);
        cache.append_token(&[1.0, 1.1], &[10.0, 10.1]).unwrap(); // t0
        cache.append_token(&[2.0, 2.1], &[20.0, 20.1]).unwrap(); // t1
        cache.append_token(&[3.0, 3.1], &[30.0, 30.1]).unwrap(); // t2

        cache.evict_prefix(1).unwrap();
        assert_eq!(cache.seq_len, 2);

        // Old t1 should now be first position (K).
        assert_eq!(cache.k[0], 2.0);
        assert_eq!(cache.k[1], 2.1);
        // Old t2 should now be second (K).
        assert_eq!(cache.k[2], 3.0);
        assert_eq!(cache.k[3], 3.1);

        // V tensor values should also be compacted correctly.
        assert_eq!(cache.v[0], 20.0);
        assert_eq!(cache.v[1], 20.1);
        assert_eq!(cache.v[2], 30.0);
        assert_eq!(cache.v[3], 30.1);
    }

    #[test]
    fn evict_prefix_zero_is_noop() {
        let mut cache = LayerKVCache::new(8, 1, 2, KVLayout::BySequence);
        cache.append_token(&[1.0, 1.1], &[10.0, 10.1]).unwrap();
        cache.evict_prefix(0).unwrap();
        assert_eq!(cache.seq_len, 1);
        assert_eq!(cache.k[0], 1.0);
    }

    #[test]
    fn evict_prefix_all_clears_cache() {
        let mut cache = LayerKVCache::new(8, 1, 2, KVLayout::BySequence);
        cache.append_token(&[1.0, 1.1], &[10.0, 10.1]).unwrap();
        cache.append_token(&[2.0, 2.1], &[20.0, 20.1]).unwrap();
        cache.evict_prefix(2).unwrap();
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn evict_prefix_by_head_layout() {
        let mut cache = LayerKVCache::new(8, 2, 2, KVLayout::ByHead);
        // t0: h0=[1,2], h1=[3,4]
        cache
            .append_token(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .unwrap();
        // t1: h0=[5,6], h1=[7,8]
        cache
            .append_token(&[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0])
            .unwrap();
        // t2: h0=[9,10], h1=[11,12]
        cache
            .append_token(&[9.0, 10.0, 11.0, 12.0], &[90.0, 100.0, 110.0, 120.0])
            .unwrap();

        cache.evict_prefix(1).unwrap();
        assert_eq!(cache.seq_len, 2);

        // Read back via read_token_at_position to verify layout-aware correctness
        let mut k_out = vec![0.0; 4];
        let mut v_out = vec![0.0; 4];
        cache.read_token_at_position(0, &mut k_out, &mut v_out);
        // Position 0 should now be old t1
        assert_eq!(k_out, [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(v_out, [50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn evict_prefix_transposed_layout() {
        let mut cache = LayerKVCache::new(8, 2, 2, KVLayout::Transposed);
        cache
            .append_token(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .unwrap();
        cache
            .append_token(&[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0])
            .unwrap();
        cache
            .append_token(&[9.0, 10.0, 11.0, 12.0], &[90.0, 100.0, 110.0, 120.0])
            .unwrap();

        cache.evict_prefix(1).unwrap();
        assert_eq!(cache.seq_len, 2);

        let mut k_out = vec![0.0; 4];
        let mut v_out = vec![0.0; 4];
        cache.read_token_at_position(0, &mut k_out, &mut v_out);
        assert_eq!(k_out, [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(v_out, [50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn sliding_window_evicts_expected_count() {
        let mut cache = LayerKVCache::new(8, 1, 2, KVLayout::BySequence);
        for i in 0..5 {
            let x = i as f32;
            cache
                .append_token(&[x, x + 0.1], &[x + 10.0, x + 10.1])
                .unwrap();
        }

        let evicted = cache.apply_sliding_window(3).unwrap();
        assert_eq!(evicted, 2);
        assert_eq!(cache.seq_len, 3);
    }

    #[test]
    fn session_sliding_window_applies_to_all_layers() {
        let mut session = SessionKVCache::new(2, 8, 1, 2, KVLayout::BySequence);
        for layer in &mut session.layers {
            for i in 0..4 {
                let x = i as f32;
                layer
                    .append_token(&[x, x + 0.1], &[x + 10.0, x + 10.1])
                    .unwrap();
            }
        }

        let evicted = session.apply_sliding_window(2).unwrap();
        assert_eq!(evicted, 2);
        for layer in &session.layers {
            assert_eq!(layer.seq_len, 2);
        }
    }

    #[test]
    fn evict_prefix_out_of_range_errors() {
        let mut cache = LayerKVCache::new(4, 1, 2, KVLayout::BySequence);
        cache.append_token(&[1.0, 1.1], &[10.0, 10.1]).unwrap();
        let err = cache.evict_prefix(2);
        assert!(matches!(
            err,
            Err(KVError::EvictionOutOfRange {
                requested: 2,
                available: 1
            })
        ));
    }

    #[test]
    #[should_panic(expected = "n_layers > 0")]
    fn session_zero_layers_panics() {
        SessionKVCache::new(0, 10, 2, 4, KVLayout::BySequence);
    }

    #[test]
    fn session_append_token_atomic_on_capacity_error() {
        let mut session = SessionKVCache::new(2, 1, 2, 4, KVLayout::BySequence);
        let k_token = vec![0.1; 8];
        let v_token = vec![0.2; 8];
        let k_tokens: Vec<&[f32]> = vec![k_token.as_slice(); 2];
        let v_tokens: Vec<&[f32]> = vec![v_token.as_slice(); 2];

        session.append_token(&k_tokens, &v_tokens).unwrap();
        assert_eq!(session.seq_len(), 1);

        // Should fail atomically — no layer should advance.
        let result = session.append_token(&k_tokens, &v_tokens);
        assert!(result.is_err());
        for i in 0..2 {
            assert_eq!(session.layer(i).unwrap().seq_len, 1);
        }
    }

    #[test]
    fn session_write_prefill_rollback_on_error() {
        let mut session = SessionKVCache::new(2, 4, 2, 4, KVLayout::BySequence);
        let k_good = vec![0.1; 32]; // 4 tokens * 2 heads * 4 dim
        let v_good = vec![0.2; 32];
        let k_bad = vec![0.1; 17]; // Wrong shape
        let v_bad = vec![0.2; 17];

        let result = session.write_prefill(
            &[k_good.as_slice(), k_bad.as_slice()],
            &[v_good.as_slice(), v_bad.as_slice()],
            4,
        );
        assert!(result.is_err());

        // After rollback, all layers should be at seq_len=0.
        for i in 0..2 {
            assert_eq!(session.layer(i).unwrap().seq_len, 0);
        }
    }

    #[test]
    fn session_write_prefill_across_layers() {
        let mut session = SessionKVCache::new(2, 10, 2, 4, KVLayout::BySequence);
        let k_seq = vec![0.1; 32]; // 4 tokens
        let v_seq = vec![0.2; 32];

        session
            .write_prefill(
                &[k_seq.as_slice(), k_seq.as_slice()],
                &[v_seq.as_slice(), v_seq.as_slice()],
                4,
            )
            .unwrap();

        assert_eq!(session.seq_len(), 4);
    }
}
