//! Tests for model configuration.

use llama::model::config::ModelConfig;

#[test]
fn test_llama3_8b_preset() {
    let config = ModelConfig::llama3_8b();
    assert_eq!(config.vocab_size, 128256);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.intermediate_size, 14336);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert!(config.uses_gqa());
    assert_eq!(config.num_queries_per_kv(), 4);
    assert!(!config.is_quantized());
}

#[test]
fn test_llama3_70b_preset() {
    let config = ModelConfig::llama3_70b();
    assert_eq!(config.vocab_size, 128256);
    assert_eq!(config.hidden_size, 8192);
    assert_eq!(config.num_hidden_layers, 80);
    assert_eq!(config.num_attention_heads, 64);
    assert_eq!(config.num_key_value_heads, 8);
    assert!(config.uses_gqa());
    assert_eq!(config.num_queries_per_kv(), 8);
}

#[test]
fn test_config_resolve() {
    let mut config = ModelConfig::llama3_8b();
    config.head_dim = 0; // Force recomputation
    config.resolve();
    assert_eq!(config.head_dim, 128); // 4096 / 32
}

#[test]
fn test_config_from_json() {
    let json = r#"{
        "vocab_size": 128256,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "max_position_embeddings": 8192
    }"#;

    let mut config: ModelConfig = serde_json::from_str(json).unwrap();
    config.resolve();

    assert_eq!(config.vocab_size, 128256);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.rope_theta, 500000.0);
}

#[test]
fn test_estimated_memory() {
    let config = ModelConfig::llama3_8b();
    let mem_fp16 = config.estimated_memory_bytes();
    // 8B model should be roughly 16GB in fp16
    assert!(mem_fp16 > 10_000_000_000); // > 10GB
    assert!(mem_fp16 < 30_000_000_000); // < 30GB
}

#[test]
fn test_quantized_config() {
    let json = r#"{
        "vocab_size": 128256,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "quantization_bits": 4,
        "quantization_group_size": 32
    }"#;

    let config: ModelConfig = serde_json::from_str(json).unwrap();
    assert!(config.is_quantized());
    assert_eq!(config.quantization_bits, 4);
    assert_eq!(config.quantization_group_size, 32);
}
