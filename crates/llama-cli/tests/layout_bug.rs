use llama_cli::{TinyModel, TinyModelConfig};
use llama_kv::{KVLayout, LayerKVCache};

#[test]
fn test_kv_layout_mismatch_bug() {
    let config = TinyModelConfig::default();
    let model = TinyModel::new(config.clone());

    // Reference with BySequence
    let mut cache_seq = LayerKVCache::new(
        config.max_seq_len,
        config.n_heads,
        config.head_dim,
        KVLayout::BySequence,
    );
    let logits_seq = model.forward_prefill(&[1, 2, 3], &mut cache_seq).unwrap();

    // With ByHead - this SHOULD produce the same logits if it were correct,
    // but attention_decode will read wrong values.
    let mut cache_head = LayerKVCache::new(
        config.max_seq_len,
        config.n_heads,
        config.head_dim,
        KVLayout::ByHead,
    );
    let logits_head = model.forward_prefill(&[1, 2, 3], &mut cache_head).unwrap();

    assert_eq!(
        logits_seq, logits_head,
        "Logits should be identical regardless of KV layout"
    );
}
