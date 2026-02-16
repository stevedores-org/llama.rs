//! Integration tests for llama-engine core trait and types.
//!
//! Validates:
//! - LlamaEngine trait can be implemented by mock backends
//! - Error types display correctly and carry context
//! - Core types satisfy required trait bounds (Send, Sync, Clone, etc.)
//! - Trait objects work for dynamic dispatch (the "narrow waist" pattern)
//! - Multiple backend implementations can coexist through the trait

use llama_engine::*;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Mock Backends
// ---------------------------------------------------------------------------

/// A simple mock engine that returns canned responses.
/// Demonstrates that the LlamaEngine trait can be implemented.
struct MockCpuEngine {
    vocab: Vec<String>,
}

impl MockCpuEngine {
    fn new() -> Self {
        Self {
            vocab: vec![
                "hello".to_string(),
                "world".to_string(),
                "the".to_string(),
                "llama".to_string(),
            ],
        }
    }
}

impl LlamaEngine for MockCpuEngine {
    fn load_model(&self, spec: &ModelSpec) -> Result<ModelHandle> {
        if spec.path.is_empty() {
            return Err(LlamaError::ModelLoad("empty path".to_string()));
        }
        Ok(ModelHandle { id: 1 })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        let tokens: Vec<i32> = text
            .split_whitespace()
            .map(|word| {
                self.vocab
                    .iter()
                    .position(|w| w == word)
                    .map(|i| i as i32)
                    .unwrap_or(-1)
            })
            .collect();

        if tokens.contains(&-1) {
            return Err(LlamaError::Tokenization("unknown token".to_string()));
        }
        Ok(tokens)
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let words: std::result::Result<Vec<&str>, _> = tokens
            .iter()
            .map(|&id| {
                self.vocab
                    .get(id as usize)
                    .map(|s| s.as_str())
                    .ok_or_else(|| LlamaError::Tokenization(format!("invalid id {}", id)))
            })
            .collect();
        Ok(words?.join(" "))
    }

    fn prefill(&self, session: &mut Session, tokens: &[i32]) -> Result<PrefillResult> {
        if tokens.is_empty() {
            return Err(LlamaError::Inference("empty prompt".to_string()));
        }
        // In a real implementation this would populate the KV cache
        let _ = session;
        Ok(PrefillResult)
    }

    fn decode(&self, session: &mut Session) -> Result<TokenStream> {
        let _ = session;
        Ok(TokenStream)
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| vec![t.len() as f32; 4]).collect())
    }
}

/// A second mock backend to demonstrate pluggability.
struct MockMetalEngine;

impl LlamaEngine for MockMetalEngine {
    fn load_model(&self, _spec: &ModelSpec) -> Result<ModelHandle> {
        Ok(ModelHandle { id: 42 })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        // Different tokenization strategy than CPU
        Ok(text.chars().map(|c| c as i32).collect())
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let s: String = tokens
            .iter()
            .filter_map(|&t| char::from_u32(t as u32))
            .collect();
        Ok(s)
    }

    fn prefill(&self, _session: &mut Session, _tokens: &[i32]) -> Result<PrefillResult> {
        Ok(PrefillResult)
    }

    fn decode(&self, _session: &mut Session) -> Result<TokenStream> {
        Ok(TokenStream)
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
    }
}

/// A failing backend that always returns errors.
struct FailingEngine;

impl LlamaEngine for FailingEngine {
    fn load_model(&self, _spec: &ModelSpec) -> Result<ModelHandle> {
        Err(LlamaError::ModelLoad("backend unavailable".to_string()))
    }

    fn tokenize(&self, _text: &str) -> Result<Vec<i32>> {
        Err(LlamaError::Tokenization("no tokenizer loaded".to_string()))
    }

    fn detokenize(&self, _tokens: &[i32]) -> Result<String> {
        Err(LlamaError::Tokenization("no tokenizer loaded".to_string()))
    }

    fn prefill(&self, _session: &mut Session, _tokens: &[i32]) -> Result<PrefillResult> {
        Err(LlamaError::Inference("no model loaded".to_string()))
    }

    fn decode(&self, _session: &mut Session) -> Result<TokenStream> {
        Err(LlamaError::Inference("no model loaded".to_string()))
    }

    fn embed(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Err(LlamaError::Inference("embedding not supported".to_string()))
    }
}

// ---------------------------------------------------------------------------
// Trait Implementation Tests
// ---------------------------------------------------------------------------

#[test]
fn mock_cpu_engine_load_model() {
    let engine = MockCpuEngine::new();
    let spec = ModelSpec {
        path: "/models/tiny-llama.safetensors".to_string(),
        context_size: 2048,
    };
    let handle = engine.load_model(&spec).unwrap();
    assert_eq!(handle.id, 1);
}

#[test]
fn mock_cpu_engine_load_model_empty_path_errors() {
    let engine = MockCpuEngine::new();
    let spec = ModelSpec {
        path: String::new(),
        context_size: 2048,
    };
    let err = engine.load_model(&spec).unwrap_err();
    assert!(matches!(err, LlamaError::ModelLoad(_)));
}

#[test]
fn mock_cpu_engine_tokenize_known_words() {
    let engine = MockCpuEngine::new();
    let tokens = engine.tokenize("hello world").unwrap();
    assert_eq!(tokens, vec![0, 1]);
}

#[test]
fn mock_cpu_engine_tokenize_unknown_word_errors() {
    let engine = MockCpuEngine::new();
    let err = engine.tokenize("hello unknown").unwrap_err();
    assert!(matches!(err, LlamaError::Tokenization(_)));
}

#[test]
fn mock_cpu_engine_tokenize_empty() {
    let engine = MockCpuEngine::new();
    let tokens = engine.tokenize("").unwrap();
    assert!(tokens.is_empty());
}

#[test]
fn mock_cpu_engine_detokenize() {
    let engine = MockCpuEngine::new();
    let text = engine.detokenize(&[0, 1]).unwrap();
    assert_eq!(text, "hello world");
}

#[test]
fn mock_cpu_engine_tokenize_detokenize_roundtrip() {
    let engine = MockCpuEngine::new();
    let original = "hello world the llama";
    let tokens = engine.tokenize(original).unwrap();
    let reconstructed = engine.detokenize(&tokens).unwrap();
    assert_eq!(reconstructed, original);
}

#[test]
fn mock_cpu_engine_prefill_empty_errors() {
    let engine = MockCpuEngine::new();
    let mut session = Session { id: 1 };
    let result = engine.prefill(&mut session, &[]);
    assert!(result.is_err());
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected error"),
    };
    assert!(matches!(err, LlamaError::Inference(_)));
}

#[test]
fn mock_cpu_engine_prefill_succeeds() {
    let engine = MockCpuEngine::new();
    let mut session = Session { id: 1 };
    assert!(engine.prefill(&mut session, &[0, 1]).is_ok());
}

#[test]
fn mock_cpu_engine_decode_succeeds() {
    let engine = MockCpuEngine::new();
    let mut session = Session { id: 1 };
    assert!(engine.decode(&mut session).is_ok());
}

#[test]
fn mock_cpu_engine_embed() {
    let engine = MockCpuEngine::new();
    let embeddings = engine.embed(&["hello", "world"]).unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), 4);
    assert_eq!(embeddings[1].len(), 4);
}

// ---------------------------------------------------------------------------
// Pluggable Backend Tests (Narrow Waist Pattern)
// ---------------------------------------------------------------------------

#[test]
fn trait_object_dispatch_cpu() {
    let engine: Box<dyn LlamaEngine> = Box::new(MockCpuEngine::new());
    let tokens = engine.tokenize("hello").unwrap();
    assert_eq!(tokens, vec![0]);
}

#[test]
fn trait_object_dispatch_metal() {
    let engine: Box<dyn LlamaEngine> = Box::new(MockMetalEngine);
    let tokens = engine.tokenize("hi").unwrap();
    // MockMetalEngine tokenizes by character code points
    assert_eq!(tokens, vec!['h' as i32, 'i' as i32]);
}

#[test]
fn different_backends_produce_different_tokenizations() {
    let cpu: Box<dyn LlamaEngine> = Box::new(MockCpuEngine::new());
    let metal: Box<dyn LlamaEngine> = Box::new(MockMetalEngine);

    let cpu_tokens = cpu.tokenize("hello").unwrap();
    let metal_tokens = metal.tokenize("hello").unwrap();

    // Different backends use different tokenization strategies
    assert_ne!(cpu_tokens, metal_tokens);
}

#[test]
fn backend_selection_at_runtime() {
    let use_metal = false;
    let engine: Box<dyn LlamaEngine> = if use_metal {
        Box::new(MockMetalEngine)
    } else {
        Box::new(MockCpuEngine::new())
    };

    // The caller code is identical regardless of backend
    let spec = ModelSpec {
        path: "/model".to_string(),
        context_size: 1024,
    };
    assert!(engine.load_model(&spec).is_ok());
}

// ---------------------------------------------------------------------------
// Error Type Tests
// ---------------------------------------------------------------------------

#[test]
fn error_model_load_display() {
    let err = LlamaError::ModelLoad("file not found".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("Model loading failed"));
    assert!(msg.contains("file not found"));
}

#[test]
fn error_tokenization_display() {
    let err = LlamaError::Tokenization("unknown token".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("Tokenization failed"));
    assert!(msg.contains("unknown token"));
}

#[test]
fn error_inference_display() {
    let err = LlamaError::Inference("OOM".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("Inference failed"));
    assert!(msg.contains("OOM"));
}

#[test]
fn error_is_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(LlamaError::ModelLoad("test".to_string()));
    assert!(err.to_string().contains("test"));
}

#[test]
fn error_debug_format() {
    let err = LlamaError::ModelLoad("debug test".to_string());
    let debug = format!("{:?}", err);
    assert!(debug.contains("ModelLoad"));
    assert!(debug.contains("debug test"));
}

#[test]
fn failing_engine_all_methods_error() {
    let engine = FailingEngine;
    let spec = ModelSpec {
        path: "/x".to_string(),
        context_size: 1,
    };
    assert!(engine.load_model(&spec).is_err());
    assert!(engine.tokenize("x").is_err());
    assert!(engine.detokenize(&[0]).is_err());
    assert!(engine.prefill(&mut Session { id: 0 }, &[0]).is_err());
    assert!(engine.decode(&mut Session { id: 0 }).is_err());
    assert!(engine.embed(&["x"]).is_err());
}

// ---------------------------------------------------------------------------
// Type Trait Bound Tests
// ---------------------------------------------------------------------------

#[test]
fn model_handle_clone_and_eq() {
    let h1 = ModelHandle { id: 42 };
    let h2 = h1.clone();
    assert_eq!(h1, h2);
}

#[test]
fn model_handle_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ModelHandle { id: 1 });
    set.insert(ModelHandle { id: 2 });
    set.insert(ModelHandle { id: 1 }); // duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn model_handle_ne() {
    let h1 = ModelHandle { id: 1 };
    let h2 = ModelHandle { id: 2 };
    assert_ne!(h1, h2);
}

#[test]
fn session_clone_and_eq() {
    let s1 = Session { id: 100 };
    let s2 = s1.clone();
    assert_eq!(s1, s2);
}

#[test]
fn session_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Session { id: 1 });
    set.insert(Session { id: 2 });
    set.insert(Session { id: 1 });
    assert_eq!(set.len(), 2);
}

#[test]
fn session_debug() {
    let s = Session { id: 42 };
    let debug = format!("{:?}", s);
    assert!(debug.contains("42"));
}

#[test]
fn model_spec_construction() {
    let spec = ModelSpec {
        path: "/models/llama-3-8b.safetensors".to_string(),
        context_size: 8192,
    };
    assert_eq!(spec.path, "/models/llama-3-8b.safetensors");
    assert_eq!(spec.context_size, 8192);
}

// ---------------------------------------------------------------------------
// Send + Sync Compile-Time Tests
// ---------------------------------------------------------------------------

/// Compile-time assertion: LlamaEngine trait requires Send + Sync.
/// This test verifies that mock implementations can be shared across threads.
#[test]
fn engine_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MockCpuEngine>();
    assert_send_sync::<MockMetalEngine>();
    assert_send_sync::<FailingEngine>();
}

#[test]
fn engine_behind_arc_is_thread_safe() {
    let engine: Arc<dyn LlamaEngine> = Arc::new(MockCpuEngine::new());
    let engine_clone = Arc::clone(&engine);

    let handle = std::thread::spawn(move || engine_clone.tokenize("hello").unwrap());

    let tokens_main = engine.tokenize("hello").unwrap();
    let tokens_thread = handle.join().unwrap();
    assert_eq!(tokens_main, tokens_thread);
}

#[test]
fn multiple_sessions_independent() {
    let engine = MockCpuEngine::new();
    let mut s1 = Session { id: 1 };
    let mut s2 = Session { id: 2 };

    // Both sessions can prefill independently
    assert!(engine.prefill(&mut s1, &[0, 1]).is_ok());
    assert!(engine.prefill(&mut s2, &[2, 3]).is_ok());

    // Sessions maintain distinct identities
    assert_ne!(s1, s2);
}

// ---------------------------------------------------------------------------
// Result Type Alias Tests
// ---------------------------------------------------------------------------

#[test]
fn result_type_ok_variant() {
    let result: Result<i32> = Ok(42);
    assert!(result.is_ok());
}

#[test]
fn result_type_err_variant() {
    let result: Result<i32> = Err(LlamaError::Inference("test".to_string()));
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Actor Pattern Simulation (Architecture Blueprint Section 8.1)
// ---------------------------------------------------------------------------

/// Simulates the "sidecar actor" pattern from the architecture blueprint.
/// The engine runs in a dedicated thread, communicating via channels.
#[test]
fn actor_pattern_channel_communication() {
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let engine = Arc::new(MockCpuEngine::new());

    let actor_engine = Arc::clone(&engine);
    let actor = std::thread::spawn(move || {
        // Simulate: receive prompt, tokenize, respond
        let prompt = "hello world";
        let tokens = actor_engine.tokenize(prompt).unwrap();
        let response = actor_engine.detokenize(&tokens).unwrap();
        tx.send(response).unwrap();
    });

    let result = rx.recv().unwrap();
    assert_eq!(result, "hello world");
    actor.join().unwrap();
}
