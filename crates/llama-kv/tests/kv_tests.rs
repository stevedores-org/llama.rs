//! Integration tests for llama-kv.
//!
//! Validates:
//! - True Test #2: KV equivalence (prefill then decode consistency)
//! - Layout equivalence (same logical data across all 3 memory layouts)
//! - Prefill + decode workflow (the standard inference pipeline)
//! - Capacity management and boundary conditions
//! - Memory accounting correctness
//! - Session-level multi-layer operations
//! - Read-back verification for all layout policies
//! - Shape tracking and display
//! - Error paths (capacity exceeded, shape mismatch, non-empty prefill)

use llama_kv::*;

// ===========================================================================
// TRUE TEST #2: KV Equivalence
// (full_forward logits == prefill(prompt[:-1]) + decode(last_token) logits)
// ===========================================================================
// Simulated at the KV cache level: writing via prefill must produce the same
// physical cache state as writing via sequential token appends.

#[test]
fn true_test_kv_equivalence_prefill_vs_sequential_append() {
    let n_heads = 4;
    let head_dim = 8;
    let seq_len = 5;
    let token_size = n_heads * head_dim;

    // Generate deterministic test data
    let k_data: Vec<f32> = (0..seq_len * token_size)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..seq_len * token_size)
        .map(|i| (i as f32) * 0.2 + 1.0)
        .collect();

    // Method 1: Prefill all at once
    let mut cache_prefill = LayerKVCache::new(16, n_heads, head_dim, KVLayout::BySequence);
    cache_prefill
        .write_prefill(&k_data, &v_data, seq_len)
        .unwrap();

    // Method 2: Append one token at a time
    let mut cache_append = LayerKVCache::new(16, n_heads, head_dim, KVLayout::BySequence);
    for seq in 0..seq_len {
        let offset = seq * token_size;
        let k_token = &k_data[offset..offset + token_size];
        let v_token = &v_data[offset..offset + token_size];
        cache_append.append_token(k_token, v_token).unwrap();
    }

    // Equivalence: both methods should produce identical cache state
    assert_eq!(cache_prefill.seq_len, cache_append.seq_len);
    assert_eq!(cache_prefill.k, cache_append.k);
    assert_eq!(cache_prefill.v, cache_append.v);
}

#[test]
fn true_test_kv_equivalence_all_layouts() {
    let n_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let token_size = n_heads * head_dim;

    let k_data: Vec<f32> = (0..seq_len * token_size)
        .map(|i| (i as f32) + 1.0)
        .collect();
    let v_data: Vec<f32> = (0..seq_len * token_size)
        .map(|i| (i as f32) * 10.0)
        .collect();

    for layout in [KVLayout::BySequence, KVLayout::ByHead, KVLayout::Transposed] {
        // Prefill
        let mut cache_prefill = LayerKVCache::new(16, n_heads, head_dim, layout);
        cache_prefill
            .write_prefill(&k_data, &v_data, seq_len)
            .unwrap();

        // Sequential append
        let mut cache_append = LayerKVCache::new(16, n_heads, head_dim, layout);
        for seq in 0..seq_len {
            let offset = seq * token_size;
            cache_append
                .append_token(
                    &k_data[offset..offset + token_size],
                    &v_data[offset..offset + token_size],
                )
                .unwrap();
        }

        assert_eq!(
            cache_prefill.k, cache_append.k,
            "K mismatch for layout {:?}",
            layout
        );
        assert_eq!(
            cache_prefill.v, cache_append.v,
            "V mismatch for layout {:?}",
            layout
        );
    }
}

// ===========================================================================
// Layout Equivalence Tests
// All three layouts should store the same logical data, just in different
// physical arrangements.
// ===========================================================================

/// Helper: read a logical value from the cache by (seq, head, dim) coordinates.
fn read_k_value(cache: &LayerKVCache, seq: usize, head: usize, dim: usize) -> f32 {
    let idx = match cache.layout {
        KVLayout::BySequence => (seq * cache.n_heads + head) * cache.head_dim + dim,
        KVLayout::ByHead => (head * cache.max_seq_len + seq) * cache.head_dim + dim,
        KVLayout::Transposed => (head * cache.head_dim + dim) * cache.max_seq_len + seq,
    };
    cache.k[idx]
}

fn read_v_value(cache: &LayerKVCache, seq: usize, head: usize, dim: usize) -> f32 {
    let idx = match cache.layout {
        KVLayout::BySequence => (seq * cache.n_heads + head) * cache.head_dim + dim,
        KVLayout::ByHead => (head * cache.max_seq_len + seq) * cache.head_dim + dim,
        KVLayout::Transposed => (head * cache.head_dim + dim) * cache.max_seq_len + seq,
    };
    cache.v[idx]
}

#[test]
fn layout_equivalence_logical_values_match() {
    let n_heads = 3;
    let head_dim = 4;
    let max_seq = 8;
    let token_size = n_heads * head_dim;

    // Write 4 tokens of known data
    let k_token_data: Vec<Vec<f32>> = (0..4)
        .map(|seq| (0..token_size).map(|i| (seq * 100 + i) as f32).collect())
        .collect();
    let v_token_data: Vec<Vec<f32>> = (0..4)
        .map(|seq| (0..token_size).map(|i| (seq * 1000 + i) as f32).collect())
        .collect();

    let layouts = [KVLayout::BySequence, KVLayout::ByHead, KVLayout::Transposed];
    let caches: Vec<LayerKVCache> = layouts
        .iter()
        .map(|&layout| {
            let mut cache = LayerKVCache::new(max_seq, n_heads, head_dim, layout);
            for seq in 0..4 {
                cache
                    .append_token(&k_token_data[seq], &v_token_data[seq])
                    .unwrap();
            }
            cache
        })
        .collect();

    // All layouts should return the same logical values
    for seq in 0..4 {
        for head in 0..n_heads {
            for dim in 0..head_dim {
                let expected_k = k_token_data[seq][head * head_dim + dim];
                let expected_v = v_token_data[seq][head * head_dim + dim];

                for (i, cache) in caches.iter().enumerate() {
                    assert_eq!(
                        read_k_value(cache, seq, head, dim),
                        expected_k,
                        "K mismatch at seq={}, head={}, dim={} for layout {:?}",
                        seq,
                        head,
                        dim,
                        layouts[i]
                    );
                    assert_eq!(
                        read_v_value(cache, seq, head, dim),
                        expected_v,
                        "V mismatch at seq={}, head={}, dim={} for layout {:?}",
                        seq,
                        head,
                        dim,
                        layouts[i]
                    );
                }
            }
        }
    }
}

// ===========================================================================
// Prefill + Decode Workflow (Architecture Section 6)
// ===========================================================================

#[test]
fn prefill_then_decode_workflow() {
    let n_heads = 4;
    let head_dim = 8;
    let max_seq = 32;
    let token_size = n_heads * head_dim;

    let mut cache = LayerKVCache::new(max_seq, n_heads, head_dim, KVLayout::BySequence);

    // Phase 1: Prefill with 5 prompt tokens
    let prompt_len = 5;
    let k_prompt: Vec<f32> = (0..prompt_len * token_size).map(|i| i as f32).collect();
    let v_prompt: Vec<f32> = (0..prompt_len * token_size).map(|i| -(i as f32)).collect();
    cache
        .write_prefill(&k_prompt, &v_prompt, prompt_len)
        .unwrap();
    assert_eq!(cache.seq_len, 5);

    // Phase 2: Decode 10 more tokens one at a time
    for step in 0..10 {
        let k_token: Vec<f32> = vec![step as f32 + 100.0; token_size];
        let v_token: Vec<f32> = vec![step as f32 + 200.0; token_size];
        cache.append_token(&k_token, &v_token).unwrap();
    }
    assert_eq!(cache.seq_len, 15);

    // Verify prefill data intact after decode appends
    for head in 0..n_heads {
        for dim in 0..head_dim {
            let expected = (head * head_dim + dim) as f32; // seq=0
            assert_eq!(
                read_k_value(&cache, 0, head, dim),
                expected,
                "prefill data corrupted at head={}, dim={}",
                head,
                dim
            );
        }
    }
}

#[test]
fn decode_tokens_written_at_correct_offset() {
    let n_heads = 2;
    let head_dim = 2;

    let mut cache = LayerKVCache::new(8, n_heads, head_dim, KVLayout::BySequence);

    // Append first token
    cache
        .append_token(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
        .unwrap();

    // Append second token
    cache
        .append_token(&[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0])
        .unwrap();

    // Verify first token at seq=0
    assert_eq!(read_k_value(&cache, 0, 0, 0), 1.0);
    assert_eq!(read_k_value(&cache, 0, 0, 1), 2.0);
    assert_eq!(read_k_value(&cache, 0, 1, 0), 3.0);
    assert_eq!(read_k_value(&cache, 0, 1, 1), 4.0);

    // Verify second token at seq=1
    assert_eq!(read_k_value(&cache, 1, 0, 0), 5.0);
    assert_eq!(read_k_value(&cache, 1, 0, 1), 6.0);
    assert_eq!(read_k_value(&cache, 1, 1, 0), 7.0);
    assert_eq!(read_k_value(&cache, 1, 1, 1), 8.0);

    // Verify V values too
    assert_eq!(read_v_value(&cache, 0, 0, 0), 10.0);
    assert_eq!(read_v_value(&cache, 1, 1, 1), 80.0);
}

// ===========================================================================
// ByHead Layout Read-back Verification
// ===========================================================================

#[test]
fn by_head_layout_physical_storage() {
    let mut cache = LayerKVCache::new(4, 2, 3, KVLayout::ByHead);
    // 2 heads, 3 dims per head
    // token = [h0d0, h0d1, h0d2, h1d0, h1d1, h1d2]
    cache
        .append_token(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        .unwrap();

    // ByHead: [heads][max_seq_len][head_dim]
    // head 0, seq 0: indices 0..3 → [1.0, 2.0, 3.0]
    assert_eq!(cache.k[0], 1.0);
    assert_eq!(cache.k[1], 2.0);
    assert_eq!(cache.k[2], 3.0);
    // head 1, seq 0: starts at head * max_seq_len * head_dim = 1 * 4 * 3 = 12
    assert_eq!(cache.k[12], 4.0);
    assert_eq!(cache.k[13], 5.0);
    assert_eq!(cache.k[14], 6.0);

    // Also check via logical read
    assert_eq!(read_k_value(&cache, 0, 0, 0), 1.0);
    assert_eq!(read_k_value(&cache, 0, 1, 2), 6.0);
}

// ===========================================================================
// Transposed Layout Read-back Verification
// ===========================================================================

#[test]
fn transposed_layout_physical_storage() {
    let mut cache = LayerKVCache::new(4, 2, 2, KVLayout::Transposed);
    // 2 heads, 2 dims
    cache
        .append_token(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
        .unwrap();

    // Transposed: [heads][head_dim][max_seq_len]
    // h0, d0, s0 = index 0
    assert_eq!(cache.k[0], 1.0);
    // h0, d1, s0 = index (0 * 2 + 1) * 4 + 0 = 4
    assert_eq!(cache.k[4], 2.0);
    // h1, d0, s0 = index (1 * 2 + 0) * 4 + 0 = 8
    assert_eq!(cache.k[8], 3.0);
    // h1, d1, s0 = index (1 * 2 + 1) * 4 + 0 = 12
    assert_eq!(cache.k[12], 4.0);

    // Add second token
    cache
        .append_token(&[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0])
        .unwrap();

    // h0, d0, s1 = index 0 * 4 + 1 = 1
    assert_eq!(cache.k[1], 5.0);
    // h1, d1, s1 = index (1 * 2 + 1) * 4 + 1 = 13
    assert_eq!(cache.k[13], 8.0);
}

// ===========================================================================
// Capacity Management
// ===========================================================================

#[test]
fn fill_to_exact_capacity() {
    let max_seq = 4;
    let mut cache = LayerKVCache::new(max_seq, 1, 2, KVLayout::BySequence);
    let k = vec![1.0; 2];
    let v = vec![2.0; 2];

    for _ in 0..max_seq {
        cache.append_token(&k, &v).unwrap();
    }
    assert_eq!(cache.seq_len, max_seq);
}

#[test]
fn exceed_capacity_by_one() {
    let max_seq = 4;
    let mut cache = LayerKVCache::new(max_seq, 1, 2, KVLayout::BySequence);
    let k = vec![1.0; 2];
    let v = vec![2.0; 2];

    for _ in 0..max_seq {
        cache.append_token(&k, &v).unwrap();
    }

    let err = cache.append_token(&k, &v).unwrap_err();
    assert_eq!(err, KVError::CapacityExceeded { seq_len: 4, max: 4 });
}

#[test]
fn prefill_exceeds_capacity() {
    let mut cache = LayerKVCache::new(4, 1, 2, KVLayout::BySequence);
    let k_seq = vec![1.0; 5 * 2]; // 5 tokens for capacity of 4
    let v_seq = vec![2.0; 5 * 2];

    let err = cache.write_prefill(&k_seq, &v_seq, 5).unwrap_err();
    assert!(matches!(err, KVError::CapacityExceeded { .. }));
}

#[test]
fn prefill_exact_capacity() {
    let mut cache = LayerKVCache::new(4, 1, 2, KVLayout::BySequence);
    let k_seq = vec![1.0; 4 * 2];
    let v_seq = vec![2.0; 4 * 2];

    cache.write_prefill(&k_seq, &v_seq, 4).unwrap();
    assert_eq!(cache.seq_len, 4);
}

// ===========================================================================
// Shape Tracking
// ===========================================================================

#[test]
fn shape_reflects_current_state() {
    let mut cache = LayerKVCache::new(256, 32, 128, KVLayout::BySequence);

    let shape0 = cache.shape();
    assert_eq!(shape0.seq_len, 0);
    assert_eq!(shape0.n_heads, 32);
    assert_eq!(shape0.head_dim, 128);

    let k = vec![1.0; 32 * 128];
    let v = vec![2.0; 32 * 128];
    cache.append_token(&k, &v).unwrap();

    let shape1 = cache.shape();
    assert_eq!(shape1.seq_len, 1);
}

#[test]
fn kv_shape_total_elements() {
    let shape = KVShape::new(100, 32, 128);
    assert_eq!(shape.total_elements(), 100 * 32 * 128);
}

#[test]
fn kv_shape_capacity_bytes() {
    let shape = KVShape::new(10, 8, 64);
    // f32 = 4 bytes
    assert_eq!(shape.capacity_bytes(4), 10 * 8 * 64 * 4);
    // f16 = 2 bytes
    assert_eq!(shape.capacity_bytes(2), 10 * 8 * 64 * 2);
}

#[test]
fn kv_shape_display() {
    let shape = KVShape::new(128, 32, 64);
    let display = format!("{}", shape);
    assert!(display.contains("128"));
    assert!(display.contains("32"));
    assert!(display.contains("64"));
}

#[test]
fn kv_shape_clone_and_eq() {
    let s1 = KVShape::new(10, 8, 64);
    let s2 = s1;
    assert_eq!(s1, s2);
}

#[test]
fn kv_shape_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(KVShape::new(10, 8, 64));
    set.insert(KVShape::new(20, 8, 64));
    set.insert(KVShape::new(10, 8, 64)); // duplicate
    assert_eq!(set.len(), 2);
}

// ===========================================================================
// Memory Accounting
// ===========================================================================

#[test]
fn memory_bytes_calculation() {
    let cache = LayerKVCache::new(256, 32, 128, KVLayout::BySequence);
    // Total allocation: 2 tensors (K + V), each max_seq_len * n_heads * head_dim * 4 bytes
    let expected = 256 * 32 * 128 * 2 * 4; // 2 tensors * 4 bytes per f32
    assert_eq!(cache.memory_bytes(), expected);
}

#[test]
fn memory_used_bytes_grows_with_seq_len() {
    let mut cache = LayerKVCache::new(256, 4, 8, KVLayout::BySequence);
    assert_eq!(cache.memory_used_bytes(), 0);

    let k = vec![1.0; 4 * 8];
    let v = vec![2.0; 4 * 8];

    cache.append_token(&k, &v).unwrap();
    // 1 seq * 4 heads * 8 dim * 8 bytes (4 for K + 4 for V)
    assert_eq!(cache.memory_used_bytes(), 4 * 8 * 8);

    cache.append_token(&k, &v).unwrap();
    assert_eq!(cache.memory_used_bytes(), 2 * 4 * 8 * 8);
}

#[test]
fn memory_used_bytes_resets_on_clear() {
    let mut cache = LayerKVCache::new(256, 4, 8, KVLayout::BySequence);
    let k = vec![1.0; 32];
    let v = vec![2.0; 32];

    cache.append_token(&k, &v).unwrap();
    assert!(cache.memory_used_bytes() > 0);

    cache.clear();
    assert_eq!(cache.memory_used_bytes(), 0);
}

#[test]
fn memory_bytes_does_not_change_with_seq_len() {
    let mut cache = LayerKVCache::new(256, 4, 8, KVLayout::BySequence);
    let initial_bytes = cache.memory_bytes();

    let k = vec![1.0; 32];
    let v = vec![2.0; 32];
    cache.append_token(&k, &v).unwrap();

    // Total allocated memory doesn't change (pre-allocated)
    assert_eq!(cache.memory_bytes(), initial_bytes);
}

// ===========================================================================
// Clear and Reuse
// ===========================================================================

#[test]
fn clear_allows_reuse() {
    let mut cache = LayerKVCache::new(4, 1, 2, KVLayout::BySequence);
    let k = vec![1.0; 2];
    let v = vec![2.0; 2];

    // Fill to capacity
    for _ in 0..4 {
        cache.append_token(&k, &v).unwrap();
    }
    assert_eq!(cache.seq_len, 4);

    // Clear and refill
    cache.clear();
    assert_eq!(cache.seq_len, 0);

    for _ in 0..4 {
        cache.append_token(&k, &v).unwrap();
    }
    assert_eq!(cache.seq_len, 4);
}

#[test]
fn clear_allows_new_prefill() {
    let mut cache = LayerKVCache::new(8, 1, 2, KVLayout::BySequence);
    let k_seq = vec![1.0; 3 * 2];
    let v_seq = vec![2.0; 3 * 2];

    cache.write_prefill(&k_seq, &v_seq, 3).unwrap();
    assert_eq!(cache.seq_len, 3);

    cache.clear();

    // Can prefill again
    cache.write_prefill(&k_seq, &v_seq, 3).unwrap();
    assert_eq!(cache.seq_len, 3);
}

// ===========================================================================
// Error Paths
// ===========================================================================

#[test]
fn prefill_on_non_empty_cache_errors() {
    let mut cache = LayerKVCache::new(8, 1, 2, KVLayout::BySequence);
    let k = vec![1.0; 2];
    let v = vec![2.0; 2];

    cache.append_token(&k, &v).unwrap();

    let k_seq = vec![1.0; 3 * 2];
    let v_seq = vec![2.0; 3 * 2];
    let err = cache.write_prefill(&k_seq, &v_seq, 3).unwrap_err();
    assert_eq!(err, KVError::NotEmpty);
}

#[test]
fn append_token_shape_mismatch_k() {
    let mut cache = LayerKVCache::new(8, 2, 4, KVLayout::BySequence);
    let k_wrong = vec![1.0; 7]; // expected 8
    let v_correct = vec![2.0; 8];

    let err = cache.append_token(&k_wrong, &v_correct).unwrap_err();
    assert_eq!(
        err,
        KVError::ShapeMismatch {
            expected: 8,
            got: 7
        }
    );
}

#[test]
fn append_token_shape_mismatch_v() {
    let mut cache = LayerKVCache::new(8, 2, 4, KVLayout::BySequence);
    let k_correct = vec![1.0; 8];
    let v_wrong = vec![2.0; 9]; // expected 8

    let err = cache.append_token(&k_correct, &v_wrong).unwrap_err();
    assert_eq!(
        err,
        KVError::ShapeMismatch {
            expected: 8,
            got: 9
        }
    );
}

#[test]
fn prefill_shape_mismatch() {
    let mut cache = LayerKVCache::new(8, 2, 4, KVLayout::BySequence);
    let k_seq = vec![1.0; 3 * 7]; // wrong: should be 3 * 8
    let v_seq = vec![2.0; 3 * 8];

    let err = cache.write_prefill(&k_seq, &v_seq, 3).unwrap_err();
    assert!(matches!(err, KVError::ShapeMismatch { .. }));
}

#[test]
fn error_display_capacity_exceeded() {
    let err = KVError::CapacityExceeded {
        seq_len: 100,
        max: 64,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("100"));
    assert!(msg.contains("64"));
}

#[test]
fn error_display_shape_mismatch() {
    let err = KVError::ShapeMismatch {
        expected: 128,
        got: 64,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("128"));
    assert!(msg.contains("64"));
}

#[test]
fn error_display_not_empty() {
    let err = KVError::NotEmpty;
    let msg = format!("{}", err);
    assert!(msg.contains("empty"));
}

#[test]
fn error_clone_and_eq() {
    let e1 = KVError::NotEmpty;
    let e2 = e1.clone();
    assert_eq!(e1, e2);
}

#[test]
fn error_is_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(KVError::NotEmpty);
    assert!(!err.to_string().is_empty());
}

// ===========================================================================
// Session-Level KV Cache
// ===========================================================================

#[test]
fn session_cache_creation() {
    let session = SessionKVCache::new(32, 256, 8, 64, KVLayout::BySequence);
    assert_eq!(session.layers.len(), 32);
    assert_eq!(session.seq_len(), 0);
}

#[test]
fn session_cache_append_across_all_layers() {
    let n_layers = 4;
    let n_heads = 2;
    let head_dim = 4;
    let mut session = SessionKVCache::new(n_layers, 16, n_heads, head_dim, KVLayout::BySequence);

    let k = vec![1.0; n_heads * head_dim];
    let v = vec![2.0; n_heads * head_dim];

    for layer in &mut session.layers {
        layer.append_token(&k, &v).unwrap();
    }

    assert_eq!(session.seq_len(), 1);

    // All layers should have same seq_len
    for layer in &session.layers {
        assert_eq!(layer.seq_len, 1);
    }
}

#[test]
fn session_cache_clear_all() {
    let n_layers = 4;
    let n_heads = 2;
    let head_dim = 4;
    let mut session = SessionKVCache::new(n_layers, 16, n_heads, head_dim, KVLayout::BySequence);

    let k = vec![1.0; n_heads * head_dim];
    let v = vec![2.0; n_heads * head_dim];

    for layer in &mut session.layers {
        layer.append_token(&k, &v).unwrap();
    }
    assert_eq!(session.seq_len(), 1);

    session.clear();
    assert_eq!(session.seq_len(), 0);

    for layer in &session.layers {
        assert_eq!(layer.seq_len, 0);
    }
}

#[test]
fn session_cache_memory_aggregation() {
    let n_layers = 4;
    let session = SessionKVCache::new(n_layers, 256, 8, 64, KVLayout::BySequence);

    let per_layer = LayerKVCache::new(256, 8, 64, KVLayout::BySequence).memory_bytes();
    assert_eq!(session.memory_bytes(), per_layer * n_layers);
}

#[test]
fn session_cache_memory_used_aggregation() {
    let n_layers = 4;
    let n_heads = 2;
    let head_dim = 4;
    let mut session = SessionKVCache::new(n_layers, 16, n_heads, head_dim, KVLayout::BySequence);

    assert_eq!(session.memory_used_bytes(), 0);

    let k = vec![1.0; n_heads * head_dim];
    let v = vec![2.0; n_heads * head_dim];

    for layer in &mut session.layers {
        layer.append_token(&k, &v).unwrap();
    }

    let per_layer_used = n_heads * head_dim * 8; // 1 seq, 8 bytes per element (K+V)
    assert_eq!(session.memory_used_bytes(), per_layer_used * n_layers);
}

#[test]
fn session_cache_empty_layers() {
    let session = SessionKVCache::new(0, 256, 8, 64, KVLayout::BySequence);
    assert_eq!(session.layers.len(), 0);
    assert_eq!(session.seq_len(), 0);
    assert_eq!(session.memory_bytes(), 0);
}

// ===========================================================================
// Realistic Dimensions
// ===========================================================================

#[test]
fn realistic_llama_3_8b_dimensions() {
    // Llama 3 8B: 32 layers, 32 heads, 128 dim, ~8k context
    // Just test that the types work at realistic scale (small seq for speed)
    let n_layers = 32;
    let n_heads = 32;
    let head_dim = 128;
    let max_seq = 64; // small for test speed

    let mut session = SessionKVCache::new(n_layers, max_seq, n_heads, head_dim, KVLayout::ByHead);

    let k = vec![0.5; n_heads * head_dim]; // 4096 elements
    let v = vec![-0.5; n_heads * head_dim];

    // Simulate 10 decode steps across all layers
    for _step in 0..10 {
        for layer in &mut session.layers {
            layer.append_token(&k, &v).unwrap();
        }
    }

    assert_eq!(session.seq_len(), 10);

    // Memory check: 32 layers * 2 tensors * max_seq(64) * 32 heads * 128 dim * 4 bytes
    let expected_total = n_layers * 2 * max_seq * n_heads * head_dim * 4;
    assert_eq!(session.memory_bytes(), expected_total);
}

// ===========================================================================
// Layout Enum Tests
// ===========================================================================

#[test]
fn layout_eq() {
    assert_eq!(KVLayout::BySequence, KVLayout::BySequence);
    assert_eq!(KVLayout::ByHead, KVLayout::ByHead);
    assert_eq!(KVLayout::Transposed, KVLayout::Transposed);
    assert_ne!(KVLayout::BySequence, KVLayout::ByHead);
    assert_ne!(KVLayout::ByHead, KVLayout::Transposed);
}

#[test]
fn layout_debug() {
    let debug = format!("{:?}", KVLayout::BySequence);
    assert!(debug.contains("BySequence"));
}

#[test]
fn layout_clone() {
    let l1 = KVLayout::Transposed;
    let l2 = l1;
    assert_eq!(l1, l2);
}

// ===========================================================================
// Property-Style Tests
// ===========================================================================

#[test]
fn seq_len_monotonically_increases_on_append() {
    let mut cache = LayerKVCache::new(100, 1, 2, KVLayout::BySequence);
    let k = vec![1.0; 2];
    let v = vec![2.0; 2];

    let mut prev_len = 0;
    for _ in 0..50 {
        cache.append_token(&k, &v).unwrap();
        assert_eq!(cache.seq_len, prev_len + 1, "seq_len should increase by 1");
        prev_len = cache.seq_len;
    }
}

#[test]
fn shape_seq_len_matches_cache_seq_len() {
    let mut cache = LayerKVCache::new(100, 4, 8, KVLayout::BySequence);
    let k = vec![1.0; 32];
    let v = vec![2.0; 32];

    for _ in 0..20 {
        assert_eq!(cache.shape().seq_len, cache.seq_len);
        cache.append_token(&k, &v).unwrap();
    }
    assert_eq!(cache.shape().seq_len, cache.seq_len);
}
