//! Tests for sampling configuration.

use llama::sampling::SamplingConfig;

#[test]
fn test_default_sampling() {
    let config = SamplingConfig::default();
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_p, 0.9);
    assert_eq!(config.top_k, 0);
    assert_eq!(config.repetition_penalty, 1.0);
}

#[test]
fn test_greedy_sampling() {
    let config = SamplingConfig::greedy();
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_p, 1.0);
    assert_eq!(config.top_k, 0);
}

#[test]
fn test_sampling_from_json() {
    let json = r#"{
        "temperature": 0.5,
        "top_k": 40,
        "top_p": 0.95,
        "repetition_penalty": 1.1
    }"#;

    let config: SamplingConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.temperature, 0.5);
    assert_eq!(config.top_k, 40);
    assert_eq!(config.top_p, 0.95);
    assert_eq!(config.repetition_penalty, 1.1);
}
