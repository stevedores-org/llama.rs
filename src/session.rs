//! High-level inference session management.
//!
//! A [`Session`] ties together the model, tokenizer, and inference engine
//! into a convenient API for interactive LLM conversations.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::engine::actor::{ActorEvent, ActorHandle};
use crate::engine::InferenceEngine;
use crate::error::{LlamaError, Result};
use crate::model::config::ModelConfig;
use crate::model::transformer::LlamaModel;
use crate::sampling::SamplingConfig;
use crate::tokenizer::{ChatMessage, Tokenizer};
use crate::weights;

/// Configuration for creating a session.
pub struct SessionConfig {
    /// Path to the model directory containing safetensors and config.json.
    pub model_path: String,

    /// Path to the tokenizer.json file.
    pub tokenizer_path: String,

    /// Maximum sequence length (context window).
    pub max_seq_len: usize,

    /// Sampling configuration.
    pub sampling: SamplingConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        SessionConfig {
            model_path: String::new(),
            tokenizer_path: String::new(),
            max_seq_len: 8192,
            sampling: SamplingConfig::default(),
        }
    }
}

/// An interactive inference session.
///
/// Manages the full lifecycle of a conversation: loading the model,
/// maintaining chat history, generating responses, and streaming tokens.
pub struct Session {
    /// The inference engine (runs on a dedicated thread via the Actor).
    actor: ActorHandle,

    /// The tokenizer for encoding/decoding.
    tokenizer: Tokenizer,

    /// Chat history for multi-turn conversations.
    messages: Vec<ChatMessage>,

    /// Sampling configuration.
    sampling: SamplingConfig,

    /// Cancellation flag.
    _is_running: Arc<AtomicBool>,
}

impl Session {
    /// Create a new session by loading a model from disk.
    ///
    /// This memory-maps the safetensors files (fast startup) and spawns
    /// the inference actor on a dedicated OS thread.
    pub fn new(config: SessionConfig) -> Result<Self> {
        // Load model configuration
        let config_path = Path::new(&config.model_path).join("config.json");
        let model_config = weights::load_config(&config_path)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(Path::new(&config.tokenizer_path))?;

        // Discover and load weight files
        let weight_files = weights::discover_weight_files(Path::new(&config.model_path))?;
        let weight_paths: Vec<&Path> = weight_files.iter().map(|p| p.as_path()).collect();
        let _weight_store = weights::WeightStore::load(&weight_paths)?;

        // Build the model from weights
        // (In a full implementation, this would wire up each layer's weights)
        let model = Self::build_model(&model_config)?;

        // Create the inference engine
        let engine = InferenceEngine::new(model, model_config, config.max_seq_len);
        let is_running = engine.running_flag();

        // Spawn the actor
        let actor = ActorHandle::spawn(engine);

        Ok(Session {
            actor,
            tokenizer,
            messages: Vec::new(),
            sampling: config.sampling,
            _is_running: is_running,
        })
    }

    /// Build the LlamaModel from configuration.
    ///
    /// In a production implementation, this would load weights from the
    /// WeightStore into each layer. This is a structural placeholder
    /// that demonstrates the wiring.
    fn build_model(_config: &ModelConfig) -> Result<LlamaModel> {
        // This would be implemented to:
        // 1. Create Embedding from weights["model.embed_tokens.weight"]
        // 2. For each layer i:
        //    a. Create Attention with q_proj, k_proj, v_proj, o_proj from weights
        //    b. Create FFN with gate_proj, up_proj, down_proj from weights
        //    c. Create RmsNorm layers from weights
        //    d. Assemble TransformerBlock
        // 3. Create final RmsNorm from weights["model.norm.weight"]
        // 4. Create lm_head from weights["lm_head.weight"] (or tied embeddings)

        Err(LlamaError::Model(
            "model building requires weight files on Apple Silicon".into(),
        ))
    }

    /// Send a user message and generate a streaming response.
    ///
    /// Returns generated token IDs via the actor's event channel.
    pub fn send_message(&mut self, content: &str) -> Result<()> {
        self.messages.push(ChatMessage::user(content));

        let prompt_tokens = self.tokenizer.encode_chat(&self.messages);
        let stop_tokens = self.tokenizer.stop_tokens();

        self.actor.generate(
            prompt_tokens,
            self.sampling.clone(),
            4096,
            stop_tokens,
        )
    }

    /// Poll for the next event from the inference actor.
    pub fn poll_event(&self) -> Option<ActorEvent> {
        self.actor.try_recv()
    }

    /// Block until the next event.
    pub fn wait_event(&self) -> Option<ActorEvent> {
        self.actor.recv()
    }

    /// Cancel the current generation.
    pub fn cancel(&self) {
        self.actor.cancel();
    }

    /// Reset the conversation (clears history and KV cache).
    pub fn reset(&mut self) -> Result<()> {
        self.messages.clear();
        self.actor.reset()
    }

    /// Add a system message to the conversation.
    pub fn set_system_prompt(&mut self, prompt: &str) {
        // Remove existing system message if present
        self.messages.retain(|m| m.role != "system");
        // Insert at the beginning
        self.messages.insert(0, ChatMessage::system(prompt));
    }

    /// Decode token IDs to text.
    pub fn decode_tokens(&self, tokens: &[i32]) -> String {
        self.tokenizer.decode(tokens)
    }

    /// Decode a single token.
    pub fn decode_token(&self, token_id: i32) -> String {
        self.tokenizer
            .decode_token(token_id)
            .unwrap_or("")
            .to_string()
    }

    /// Get the current chat history.
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Whether generation is currently active.
    pub fn is_generating(&self) -> bool {
        self.actor.is_running()
    }

    /// Update sampling configuration.
    pub fn set_sampling(&mut self, config: SamplingConfig) {
        self.sampling = config;
    }
}
