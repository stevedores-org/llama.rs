//! Tokenizer integration for Llama models.
//!
//! Wraps the BPE tokenizer (loaded from tokenizer.json) and provides
//! chat template formatting for Llama 3's instruction format.
//!
//! Llama 3 uses a 128k-token BPE vocabulary with special control tokens
//! for structuring chat interactions.

use std::path::Path;

use crate::error::{LlamaError, Result};

/// Special tokens used by Llama 3.
pub mod special_tokens {
    pub const BOS: &str = "<|begin_of_text|>";
    pub const EOS: &str = "<|end_of_text|>";
    pub const START_HEADER: &str = "<|start_header_id|>";
    pub const END_HEADER: &str = "<|end_header_id|>";
    pub const EOT: &str = "<|eot_id|>";

    /// Default BOS token ID for Llama 3.
    pub const BOS_ID: i32 = 128000;

    /// Default EOS token ID for Llama 3.
    pub const EOS_ID: i32 = 128001;

    /// Default EOT token ID for Llama 3.
    pub const EOT_ID: i32 = 128009;
}

/// A chat message with role and content.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// Tokenizer wrapper that handles encoding, decoding, and chat formatting.
pub struct Tokenizer {
    /// Vocabulary: token_id -> token string.
    vocab: Vec<String>,

    /// Reverse vocabulary: token string -> token_id.
    token_to_id: std::collections::HashMap<String, i32>,

    /// BPE merge rules (pairs of token strings).
    merges: Vec<(String, String)>,

    /// Special token IDs.
    pub bos_id: i32,
    pub eos_id: i32,
    pub eot_id: i32,
}

impl Tokenizer {
    /// Load a tokenizer from a tokenizer.json file.
    ///
    /// The file follows the HuggingFace tokenizers format.
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).map_err(|e| {
            LlamaError::Tokenizer(format!("failed to read {}: {e}", path.display()))
        })?;

        let json: serde_json::Value = serde_json::from_str(&data).map_err(|e| {
            LlamaError::Tokenizer(format!("failed to parse tokenizer JSON: {e}"))
        })?;

        // Extract vocabulary
        let mut vocab = Vec::new();
        let mut token_to_id = std::collections::HashMap::new();

        if let Some(model) = json.get("model") {
            if let Some(vocab_obj) = model.get("vocab").and_then(|v| v.as_object()) {
                // Pre-allocate
                vocab.resize(vocab_obj.len(), String::new());

                for (token, id_val) in vocab_obj {
                    if let Some(id) = id_val.as_i64() {
                        let id = id as i32;
                        if (id as usize) < vocab.len() {
                            vocab[id as usize] = token.clone();
                        }
                        token_to_id.insert(token.clone(), id);
                    }
                }
            }
        }

        // Extract merges
        let mut merges = Vec::new();
        if let Some(model) = json.get("model") {
            if let Some(merge_list) = model.get("merges").and_then(|m| m.as_array()) {
                for merge in merge_list {
                    if let Some(s) = merge.as_str() {
                        if let Some((a, b)) = s.split_once(' ') {
                            merges.push((a.to_string(), b.to_string()));
                        }
                    }
                }
            }
        }

        // Look up special token IDs
        let bos_id = token_to_id
            .get(special_tokens::BOS)
            .copied()
            .unwrap_or(special_tokens::BOS_ID);
        let eos_id = token_to_id
            .get(special_tokens::EOS)
            .copied()
            .unwrap_or(special_tokens::EOS_ID);
        let eot_id = token_to_id
            .get(special_tokens::EOT)
            .copied()
            .unwrap_or(special_tokens::EOT_ID);

        Ok(Tokenizer {
            vocab,
            token_to_id,
            merges,
            bos_id,
            eos_id,
            eot_id,
        })
    }

    /// Encode text into token IDs.
    ///
    /// This implements a simplified BPE encoding. For production use,
    /// the full HuggingFace tokenizers library should be used.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        // Start with individual bytes/characters as initial tokens
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply BPE merges greedily
        for (a, b) in &self.merges {
            let merged = format!("{a}{b}");
            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == *a && tokens[i + 1] == *b {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        // Convert tokens to IDs
        tokens
            .iter()
            .filter_map(|t| self.token_to_id.get(t).copied())
            .collect()
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, token_ids: &[i32]) -> String {
        token_ids
            .iter()
            .filter_map(|&id| self.vocab.get(id as usize))
            .cloned()
            .collect()
    }

    /// Decode a single token ID.
    pub fn decode_token(&self, token_id: i32) -> Option<&str> {
        self.vocab.get(token_id as usize).map(|s| s.as_str())
    }

    /// Format chat messages using the Llama 3 chat template.
    ///
    /// Produces the format:
    /// ```text
    /// <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    ///
    /// {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
    ///
    /// {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ///
    /// ```
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> String {
        let mut output = String::new();
        output.push_str(special_tokens::BOS);

        for msg in messages {
            output.push_str(special_tokens::START_HEADER);
            output.push_str(&msg.role);
            output.push_str(special_tokens::END_HEADER);
            output.push_str("\n\n");
            output.push_str(&msg.content);
            output.push_str(special_tokens::EOT);
        }

        // Add the assistant header to prompt generation
        output.push_str(special_tokens::START_HEADER);
        output.push_str("assistant");
        output.push_str(special_tokens::END_HEADER);
        output.push_str("\n\n");

        output
    }

    /// Encode a chat conversation into token IDs.
    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Vec<i32> {
        let formatted = self.apply_chat_template(messages);
        self.encode(&formatted)
    }

    /// Get the stop token IDs (tokens that signal end of generation).
    pub fn stop_tokens(&self) -> Vec<i32> {
        vec![self.eos_id, self.eot_id]
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
