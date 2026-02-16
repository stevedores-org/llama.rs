//! Cross-crate integration tests: end-to-end inference pipeline.
//!
//! Validates:
//! - A mock LlamaEngine implementation that wires together tokenizer, sampling,
//!   and KV cache into a complete inference pipeline
//! - The "narrow waist" pattern works in practice with real crate dependencies
//! - Prefill → Decode → Sample → Detokenize workflow
//! - Cancellation safety (via AtomicBool, per Architecture Section 8.3)
//! - Actor pattern simulation with channels
//! - Streaming token emission matching batch output

use llama_engine::*;
use llama_kv::{KVLayout, LayerKVCache, SessionKVCache};
use llama_sampling::Sampler;
use llama_tokenizer::{DecodingState, Tokenizer, WhitespaceTokenizer};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// ===========================================================================
// Integrated Mock Engine
// ===========================================================================

/// A mock engine that wires together tokenizer, sampler, and KV cache
/// to demonstrate the full Phase 1 pipeline.
struct IntegratedMockEngine {
    tokenizer: WhitespaceTokenizer,
    n_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
}

impl IntegratedMockEngine {
    fn new() -> Self {
        Self {
            tokenizer: WhitespaceTokenizer::new(),
            n_heads: 2,
            head_dim: 4,
            max_seq_len: 64,
        }
    }

    /// Simulate a forward pass: given token IDs, produce logits.
    /// This is a trivial hash-based simulation (not a real model).
    fn mock_forward(&self, token_ids: &[i32]) -> Vec<f32> {
        let vocab_size = 10;
        let mut logits = vec![0.0f32; vocab_size];

        // Simple deterministic "model": hash the input tokens to produce logits
        for (i, logit) in logits.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for &tid in token_ids {
                acc += ((tid as f32 + 1.0) * (i as f32 + 1.0)).sin();
            }
            *logit = acc;
        }
        logits
    }

    /// Create a KV cache for a single layer.
    fn create_kv_cache(&self) -> LayerKVCache {
        LayerKVCache::new(
            self.max_seq_len,
            self.n_heads,
            self.head_dim,
            KVLayout::BySequence,
        )
    }

    /// Simulate generating fake KV data for a token.
    fn mock_kv_for_token(&self, token_id: i32) -> (Vec<f32>, Vec<f32>) {
        let size = self.n_heads * self.head_dim;
        let k: Vec<f32> = (0..size)
            .map(|i| (token_id as f32 + i as f32) * 0.01)
            .collect();
        let v: Vec<f32> = (0..size)
            .map(|i| (token_id as f32 - i as f32) * 0.01)
            .collect();
        (k, v)
    }
}

impl LlamaEngine for IntegratedMockEngine {
    fn load_model(&self, spec: &ModelSpec) -> Result<ModelHandle> {
        if spec.path.is_empty() {
            return Err(LlamaError::ModelLoad("empty path".to_string()));
        }
        Ok(ModelHandle { id: 1 })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenizer
            .encode(text)
            .map_err(|e| LlamaError::Tokenization(e.to_string()))
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.tokenizer
            .decode(tokens)
            .map_err(|e| LlamaError::Tokenization(e.to_string()))
    }

    fn prefill(&self, _session: &mut Session, tokens: &[i32]) -> Result<PrefillResult> {
        if tokens.is_empty() {
            return Err(LlamaError::Inference("empty prompt".to_string()));
        }
        Ok(PrefillResult)
    }

    fn decode(&self, _session: &mut Session) -> Result<TokenStream> {
        Ok(TokenStream)
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| vec![t.len() as f32; 4]).collect())
    }
}

// ===========================================================================
// End-to-End Pipeline Tests
// ===========================================================================

#[test]
fn pipeline_tokenize_prefill_decode_detokenize() {
    let engine = IntegratedMockEngine::new();

    // Step 1: Tokenize input
    let prompt = "hello world";
    let token_ids = engine.tokenize(prompt).unwrap();
    assert_eq!(token_ids.len(), 2);

    // Step 2: Create KV cache and simulate prefill
    let mut kv_cache = engine.create_kv_cache();
    for &tid in &token_ids {
        let (k, v) = engine.mock_kv_for_token(tid);
        kv_cache.append_token(&k, &v).unwrap();
    }
    assert_eq!(kv_cache.seq_len, token_ids.len());

    // Step 3: Forward pass to get logits
    let logits = engine.mock_forward(&token_ids);
    assert_eq!(logits.len(), 10);

    // Step 4: Sample next token
    let mut sampler = Sampler::new().with_temperature(0.8).with_seed(42);
    let next_token = sampler.sample(&logits).unwrap();
    assert!(next_token < 10);

    // Step 5: Append to KV cache (decode step)
    let (k, v) = engine.mock_kv_for_token(next_token as i32);
    kv_cache.append_token(&k, &v).unwrap();
    assert_eq!(kv_cache.seq_len, token_ids.len() + 1);
}

#[test]
fn pipeline_multi_step_generation() {
    let engine = IntegratedMockEngine::new();
    let prompt = "hello world test";
    let mut token_ids = engine.tokenize(prompt).unwrap();

    let mut kv_cache = engine.create_kv_cache();
    let mut sampler = Sampler::new()
        .with_temperature(0.7)
        .with_top_k(5)
        .with_seed(42);

    // Prefill
    for &tid in &token_ids {
        let (k, v) = engine.mock_kv_for_token(tid);
        kv_cache.append_token(&k, &v).unwrap();
    }

    // Generate 10 tokens
    let mut generated = Vec::new();
    for _ in 0..10 {
        let logits = engine.mock_forward(&token_ids);
        let next = sampler.sample(&logits).unwrap();
        generated.push(next as i32);

        let (k, v) = engine.mock_kv_for_token(next as i32);
        kv_cache.append_token(&k, &v).unwrap();

        token_ids.push(next as i32);
    }

    assert_eq!(generated.len(), 10);
    assert_eq!(kv_cache.seq_len, 3 + 10); // 3 prompt + 10 generated
}

#[test]
fn pipeline_deterministic_generation() {
    let engine = IntegratedMockEngine::new();
    let prompt = "hello world";

    let generate = |seed: u64| -> Vec<usize> {
        let mut token_ids = engine.tokenize(prompt).unwrap();
        let mut sampler = Sampler::new().with_temperature(0.8).with_seed(seed);
        let mut generated = Vec::new();

        for _ in 0..20 {
            let logits = engine.mock_forward(&token_ids);
            let next = sampler.sample(&logits).unwrap();
            generated.push(next);
            token_ids.push(next as i32);
        }

        generated
    };

    // Same seed → same output
    let seq1 = generate(42);
    let seq2 = generate(42);
    assert_eq!(seq1, seq2, "deterministic generation broken");

    // Different seed → different output
    let seq3 = generate(123);
    assert_ne!(
        seq1, seq3,
        "different seeds should produce different output"
    );
}

// ===========================================================================
// Streaming Token Emission (Architecture Section 8.2)
// ===========================================================================

#[test]
fn pipeline_streaming_matches_batch_decode() {
    let engine = IntegratedMockEngine::new();
    let prompt = "hello world test foo bar";

    // Encode the prompt
    let token_ids = engine.tokenize(prompt).unwrap();

    // Batch decode
    let batch_result = engine.detokenize(&token_ids).unwrap();

    // Streaming decode
    let mut state = DecodingState::new();
    let mut streamed = String::new();
    for &tid in &token_ids {
        let chunk = engine.tokenizer.decode_token(tid, &mut state).unwrap();
        streamed.push_str(&chunk);
    }

    assert_eq!(batch_result, streamed, "streaming must match batch");
}

// ===========================================================================
// Cancellation Safety (Architecture Section 8.3)
// ===========================================================================

#[test]
fn pipeline_cancellation_via_atomic_bool() {
    let engine = Arc::new(IntegratedMockEngine::new());
    let is_running = Arc::new(AtomicBool::new(true));

    let engine_clone = Arc::clone(&engine);
    let running_clone = Arc::clone(&is_running);

    let handle = std::thread::spawn(move || {
        let mut token_ids = engine_clone.tokenize("hello world").unwrap();
        let mut sampler = Sampler::new().with_seed(42);
        let mut steps = 0;

        // Generation loop with cancellation check
        while running_clone.load(Ordering::Relaxed) {
            let logits = engine_clone.mock_forward(&token_ids);
            let next = sampler.sample(&logits).unwrap();
            token_ids.push(next as i32);
            steps += 1;

            if steps > 1000 {
                // Safety: don't run forever in case cancellation fails
                break;
            }
        }

        steps
    });

    // Let it run briefly, then cancel
    std::thread::sleep(std::time::Duration::from_millis(10));
    is_running.store(false, Ordering::Relaxed);

    let steps = handle.join().unwrap();
    assert!(steps > 0, "should have generated at least 1 token");
    assert!(steps < 1000, "cancellation should have stopped generation");
}

#[test]
fn pipeline_cancellation_preserves_kv_state() {
    let engine = IntegratedMockEngine::new();
    let mut kv_cache = engine.create_kv_cache();
    let is_running = AtomicBool::new(true);

    let mut token_ids = engine.tokenize("hello world").unwrap();
    let mut sampler = Sampler::new().with_seed(42);

    // Prefill
    for &tid in &token_ids {
        let (k, v) = engine.mock_kv_for_token(tid);
        kv_cache.append_token(&k, &v).unwrap();
    }
    let prefill_len = kv_cache.seq_len;

    // Generate a few tokens, then cancel
    let mut steps = 0;
    while is_running.load(Ordering::Relaxed) && steps < 5 {
        let logits = engine.mock_forward(&token_ids);
        let next = sampler.sample(&logits).unwrap();
        let (k, v) = engine.mock_kv_for_token(next as i32);
        kv_cache.append_token(&k, &v).unwrap();
        token_ids.push(next as i32);
        steps += 1;

        if steps >= 3 {
            is_running.store(false, Ordering::Relaxed);
        }
    }

    // KV cache should be in a consistent state
    assert_eq!(kv_cache.seq_len, prefill_len + steps);
    assert!(kv_cache.seq_len > 0);
}

// ===========================================================================
// Actor Pattern (Architecture Section 8.1)
// ===========================================================================

#[test]
fn pipeline_actor_pattern_with_channel() {
    let (prompt_tx, prompt_rx) = std::sync::mpsc::channel::<String>();
    let (token_tx, token_rx) = std::sync::mpsc::channel::<String>();

    let engine = Arc::new(IntegratedMockEngine::new());
    let actor_engine = Arc::clone(&engine);

    // Spawn inference actor
    let actor = std::thread::spawn(move || {
        while let Ok(prompt) = prompt_rx.recv() {
            let token_ids = actor_engine.tokenize(&prompt).unwrap();
            let mut sampler = Sampler::new().with_seed(42);
            let logits = actor_engine.mock_forward(&token_ids);
            let next = sampler.sample(&logits).unwrap();

            // In a real system, we'd detokenize. Here just send the index.
            token_tx.send(format!("token:{}", next)).unwrap();
        }
    });

    // Send prompts from "main thread"
    prompt_tx.send("hello world".to_string()).unwrap();
    let response = token_rx.recv().unwrap();
    assert!(response.starts_with("token:"));

    prompt_tx.send("test input".to_string()).unwrap();
    let response2 = token_rx.recv().unwrap();
    assert!(response2.starts_with("token:"));

    drop(prompt_tx); // close channel → actor exits
    actor.join().unwrap();
}

// ===========================================================================
// Session KV Cache with Generation Pipeline
// ===========================================================================

#[test]
fn pipeline_multi_layer_kv_cache_generation() {
    let engine = IntegratedMockEngine::new();
    let n_layers = 4;
    let mut session = SessionKVCache::new(
        n_layers,
        engine.max_seq_len,
        engine.n_heads,
        engine.head_dim,
        KVLayout::BySequence,
    );

    let token_ids = engine.tokenize("hello world").unwrap();

    // Prefill all layers
    for &tid in &token_ids {
        let (k, v) = engine.mock_kv_for_token(tid);
        for layer in &mut session.layers {
            layer.append_token(&k, &v).unwrap();
        }
    }

    assert_eq!(session.seq_len(), 2);

    // Decode 5 more tokens
    let mut sampler = Sampler::new().with_seed(42);
    let mut current_tokens = token_ids;

    for _ in 0..5 {
        let logits = engine.mock_forward(&current_tokens);
        let next = sampler.sample(&logits).unwrap();

        let (k, v) = engine.mock_kv_for_token(next as i32);
        for layer in &mut session.layers {
            layer.append_token(&k, &v).unwrap();
        }

        current_tokens.push(next as i32);
    }

    assert_eq!(session.seq_len(), 7); // 2 prefill + 5 decode
    assert!(session.memory_used_bytes() > 0);
}

#[test]
fn pipeline_session_clear_and_restart() {
    let engine = IntegratedMockEngine::new();
    let mut session = SessionKVCache::new(2, 32, engine.n_heads, engine.head_dim, KVLayout::ByHead);

    // First conversation
    let tokens = engine.tokenize("hello").unwrap();
    let (k, v) = engine.mock_kv_for_token(tokens[0]);
    for layer in &mut session.layers {
        layer.append_token(&k, &v).unwrap();
    }
    assert_eq!(session.seq_len(), 1);

    // Clear for new conversation
    session.clear();
    assert_eq!(session.seq_len(), 0);

    // Second conversation
    let tokens2 = engine.tokenize("goodbye world").unwrap();
    for &tid in &tokens2 {
        let (k, v) = engine.mock_kv_for_token(tid);
        for layer in &mut session.layers {
            layer.append_token(&k, &v).unwrap();
        }
    }
    assert_eq!(session.seq_len(), 2);
}

// ===========================================================================
// Repetition Penalty Integration
// ===========================================================================

#[test]
fn pipeline_repetition_penalty_reduces_repeats() {
    let engine = IntegratedMockEngine::new();
    let token_ids = engine.tokenize("hello world").unwrap();

    // Generate without penalty
    let mut sampler_no_penalty = Sampler::new().with_seed(42);
    let mut seq_no_penalty = Vec::new();
    let mut current = token_ids.clone();
    for _ in 0..20 {
        let logits = engine.mock_forward(&current);
        let next = sampler_no_penalty.sample(&logits).unwrap();
        seq_no_penalty.push(next);
        current.push(next as i32);
    }

    // Generate with repetition penalty
    let mut sampler_penalty = Sampler::new().with_repetition_penalty(2.0).with_seed(42);
    let mut seq_penalty = Vec::new();
    let mut current = token_ids;
    let mut history = Vec::new();
    for _ in 0..20 {
        let logits = engine.mock_forward(&current);
        let next = sampler_penalty
            .sample_with_history(&logits, &history)
            .unwrap();
        seq_penalty.push(next);
        history.push(next);
        current.push(next as i32);
    }

    // With penalty, we should see more unique tokens
    let unique_no_penalty: std::collections::HashSet<_> = seq_no_penalty.iter().collect();
    let unique_penalty: std::collections::HashSet<_> = seq_penalty.iter().collect();

    assert!(
        unique_penalty.len() >= unique_no_penalty.len(),
        "penalty should encourage diversity: {} unique without, {} with",
        unique_no_penalty.len(),
        unique_penalty.len()
    );
}

// ===========================================================================
// Thread Safety of Full Pipeline
// ===========================================================================

#[test]
fn pipeline_engine_shared_across_threads() {
    let engine = Arc::new(IntegratedMockEngine::new());

    let mut handles = vec![];
    for i in 0..4 {
        let e = Arc::clone(&engine);
        handles.push(std::thread::spawn(move || {
            let prompt = format!("thread{} says hello", i);
            let tokens = e.tokenize(&prompt).unwrap();
            let logits = e.mock_forward(&tokens);
            let mut sampler = Sampler::new().with_seed(i as u64 + 1);
            sampler.sample(&logits).unwrap()
        }));
    }

    let results: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.len(), 4);
    for &r in &results {
        assert!(r < 10);
    }
}
