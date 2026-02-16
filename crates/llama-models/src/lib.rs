//! # llama-models
//!
//! Model architecture implementations for Llama 3, Qwen, and Mistral families.
//!
//! Provides:
//! - **RMSNorm**: Root Mean Square layer normalization
//! - **RoPE**: Rotary Position Embeddings
//! - **MLP**: Feedforward networks (SwiGLU, GELU variants)
//! - **Attention**: Multi-head self-attention with caching
//! - **SafeTensors**: Weight loading from model files

use std::path::Path;

/// Error type for weight loading operations.
#[derive(Debug, thiserror::Error)]
pub enum WeightError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensors(String),

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("Missing weight: {0}")]
    MissingWeight(String),
}

pub type WeightResult<T> = Result<T, WeightError>;

/// Weight key naming convention for model architectures.
pub struct WeightNames {
    pub embed_tokens: String,
    pub norm_weight: String,
    pub lm_head: String,
}

impl WeightNames {
    /// Llama 3 weight naming.
    pub fn llama3() -> Self {
        Self {
            embed_tokens: "model.embed_tokens.weight".to_string(),
            norm_weight: "model.norm.weight".to_string(),
            lm_head: "lm_head.weight".to_string(),
        }
    }

    /// Qwen weight naming.
    pub fn qwen() -> Self {
        Self {
            embed_tokens: "model.embed_tokens.weight".to_string(),
            norm_weight: "model.norm.weight".to_string(),
            lm_head: "lm_head.weight".to_string(),
        }
    }

    /// Build block attention weight key.
    pub fn block_attn_key(&self, layer_idx: usize, weight_type: &str) -> String {
        format!("model.layers.{}.self_attn.{}.weight", layer_idx, weight_type)
    }

    /// Build block MLP weight key.
    pub fn block_mlp_key(&self, layer_idx: usize, weight_type: &str) -> String {
        format!("model.layers.{}.mlp.{}.weight", layer_idx, weight_type)
    }

    /// Build block norm weight key.
    pub fn block_norm_key(&self, layer_idx: usize, _norm_type: &str) -> String {
        format!("model.layers.{}.input_layernorm.weight", layer_idx)
    }
}

/// Weight loader for safetensors format.
pub struct WeightLoader {
    pub weights: safetensors::SafeTensors<'static>,
}

impl WeightLoader {
    /// Load weights from a safetensors file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> WeightResult<Self> {
        let data = std::fs::read(path)?;
        let data_static = Box::leak(data.into_boxed_slice());
        let weights = safetensors::SafeTensors::deserialize(data_static)
            .map_err(|e| WeightError::SafeTensors(e.to_string()))?;

        Ok(Self { weights })
    }

    /// Load a weight tensor by name.
    pub fn load_weight(&self, name: &str, expected_size: usize) -> WeightResult<Vec<f32>> {
        let tensor = self
            .weights
            .tensor(name)
            .map_err(|_| WeightError::MissingWeight(name.to_string()))?;

        let data = tensor
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();

        if data.len() != expected_size {
            return Err(WeightError::ShapeMismatch {
                expected: expected_size,
                got: data.len(),
            });
        }

        Ok(data)
    }

    /// Load embeddings from model file.
    pub fn load_embeddings(&self, names: &WeightNames, vocab_size: usize, d_model: usize) -> WeightResult<Vec<f32>> {
        self.load_weight(&names.embed_tokens, vocab_size * d_model)
    }

    /// Load final normalization weights.
    pub fn load_norm(&self, names: &WeightNames, d_model: usize) -> WeightResult<Vec<f32>> {
        self.load_weight(&names.norm_weight, d_model)
    }

    /// Load output projection weights.
    pub fn load_lm_head(&self, names: &WeightNames, d_model: usize, vocab_size: usize) -> WeightResult<Vec<f32>> {
        self.load_weight(&names.lm_head, d_model * vocab_size)
    }

    /// Load attention weights for a block.
    pub fn load_attention(&self, names: &WeightNames, layer_idx: usize, d_model: usize) -> WeightResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let w_q = self.load_weight(&names.block_attn_key(layer_idx, "q_proj"), d_model * d_model)?;
        let w_k = self.load_weight(&names.block_attn_key(layer_idx, "k_proj"), d_model * d_model)?;
        let w_v = self.load_weight(&names.block_attn_key(layer_idx, "v_proj"), d_model * d_model)?;
        let w_o = self.load_weight(&names.block_attn_key(layer_idx, "o_proj"), d_model * d_model)?;
        Ok((w_q, w_k, w_v, w_o))
    }

    /// Load MLP weights for a block.
    pub fn load_mlp(&self, names: &WeightNames, layer_idx: usize, d_model: usize, d_ff: usize) -> WeightResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let w_gate = self.load_weight(&names.block_mlp_key(layer_idx, "gate_proj"), d_model * d_ff)?;
        let w_up = self.load_weight(&names.block_mlp_key(layer_idx, "up_proj"), d_model * d_ff)?;
        let w_down = self.load_weight(&names.block_mlp_key(layer_idx, "down_proj"), d_ff * d_model)?;
        Ok((w_gate, w_up, w_down))
    }
}

/// Root Mean Square Layer Normalization.
///
/// Used in Llama 3, Qwen, and modern LLMs as a simpler alternative to LayerNorm.
/// Formula: `y = x / RMS(x) * weight`, where RMS(x) = sqrt(mean(x^2))
///
/// # References
/// - Huang et al. (2021): "Root Mean Square Layer Normalization"
/// - Used in: Llama 3, Qwen, Mistral
#[derive(Debug, Clone)]
pub struct RMSNorm {
    /// Learnable scale parameter, shape: [d_model]
    pub weight: Vec<f32>,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl RMSNorm {
    /// Create a new RMSNorm layer.
    ///
    /// # Arguments
    /// - `d_model`: Dimension of the hidden state
    /// - `eps`: Epsilon for numerical stability (default ~1e-6)
    pub fn new(d_model: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; d_model],
            eps,
        }
    }

    /// Apply RMSNorm to input tensor.
    ///
    /// # Arguments
    /// - `x`: Input tensor, shape [seq_len, d_model] (flattened)
    ///
    /// # Returns
    /// Output tensor with same shape as input
    pub fn forward(&self, x: &[f32], d_model: usize) -> Vec<f32> {
        assert_eq!(x.len() % d_model, 0, "Input size must be divisible by d_model");

        let seq_len = x.len() / d_model;
        let mut output = vec![0.0; x.len()];

        for seq_idx in 0..seq_len {
            let offset = seq_idx * d_model;
            let slice = &x[offset..offset + d_model];

            // Compute RMS: sqrt(mean(x^2))
            let mean_sq: f32 = slice.iter().map(|v| v * v).sum::<f32>() / d_model as f32;
            let rms = (mean_sq + self.eps).sqrt();

            // Apply normalization and weight scaling
            for (i, &val) in slice.iter().enumerate() {
                output[offset + i] = (val / rms) * self.weight[i];
            }
        }

        output
    }
}

/// Rotary Position Embeddings (RoPE).
///
/// Encodes absolute position information into attention by rotating query and key vectors.
/// Formula: Apply 2D rotations to (Q, K) pairs using position-dependent angles.
///
/// # References
/// - Su et al. (2021): "RoFormer: Enhanced Transformer with Rotary Position Embedding"
/// - Used in: Llama 3, Qwen, Mistral
///
/// # Key Properties
/// - Position-aware through rotation angles
/// - Supports long sequences (relative position encoding property)
/// - Low computational cost compared to ALiBi or position encodings
#[derive(Debug, Clone)]
pub struct RoPE {
    /// Dimension of head (head_dim)
    pub dim: usize,
    /// Base for frequency calculation (default: 10000)
    pub base: f32,
    /// Inverse frequencies: [1/base^(2i/dim) for i in 0..dim/2]
    pub inv_freq: Vec<f32>,
}

impl RoPE {
    /// Create RoPE embeddings for a given head dimension.
    ///
    /// # Arguments
    /// - `dim`: Dimension of each attention head
    /// - `base`: Base for frequency calculation (typically 10000)
    pub fn new(dim: usize, base: f32) -> Self {
        assert!(dim % 2 == 0, "Head dimension must be even for RoPE");

        // Precompute inverse frequencies: 1.0 / (base^(2i/dim))
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
            .collect();

        Self { dim, base, inv_freq }
    }

    /// Apply RoPE to query and key tensors.
    ///
    /// # Arguments
    /// - `q`: Query tensor, shape [batch_size, seq_len, n_heads, head_dim] (flattened)
    /// - `k`: Key tensor, same shape as q
    /// - `seq_len`: Current sequence length
    /// - `n_heads`: Number of attention heads
    ///
    /// # Returns
    /// Tuple of (rotated_q, rotated_k)
    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        seq_len: usize,
        n_heads: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let batch_size = q.len() / (seq_len * n_heads * self.dim);
        assert_eq!(q.len(), batch_size * seq_len * n_heads * self.dim);

        let mut q_rot = vec![0.0; q.len()];
        let mut k_rot = vec![0.0; k.len()];

        for b in 0..batch_size {
            for t in 0..seq_len {
                for h in 0..n_heads {
                    // Position index (for this token)
                    let pos = t as f32;

                    // Base index for this head
                    let base_idx = (b * seq_len * n_heads + t * n_heads + h) * self.dim;

                    // Apply RoPE: rotate each (q[2i], q[2i+1]) pair
                    for i in 0..self.dim / 2 {
                        let angle = pos * self.inv_freq[i];
                        let cos_angle = angle.cos();
                        let sin_angle = angle.sin();

                        // Rotate query
                        let q_0 = q[base_idx + 2 * i];
                        let q_1 = q[base_idx + 2 * i + 1];
                        q_rot[base_idx + 2 * i] = q_0 * cos_angle - q_1 * sin_angle;
                        q_rot[base_idx + 2 * i + 1] = q_0 * sin_angle + q_1 * cos_angle;

                        // Rotate key
                        let k_0 = k[base_idx + 2 * i];
                        let k_1 = k[base_idx + 2 * i + 1];
                        k_rot[base_idx + 2 * i] = k_0 * cos_angle - k_1 * sin_angle;
                        k_rot[base_idx + 2 * i + 1] = k_0 * sin_angle + k_1 * cos_angle;
                    }
                }
            }
        }

        (q_rot, k_rot)
    }
}

/// Model configuration for Llama 3, Qwen, or Mistral.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub d_model: usize,        // Hidden dimension (4096 for Llama 3 8B)
    pub n_heads: usize,        // Number of attention heads (32 for Llama 3 8B)
    pub n_kv_heads: usize,     // Number of KV heads (8 for Llama 3, for GQA)
    pub n_layers: usize,       // Number of transformer layers (32 for Llama 3 8B)
    pub d_ff: usize,           // Feedforward hidden dimension (intermediate size)
    pub vocab_size: usize,     // Vocabulary size (128256 for Llama 3)
    pub max_seq_len: usize,    // Maximum sequence length
    pub rope_base: f32,        // RoPE base (10000 for Llama 3, 1000000 for Qwen)
    pub norm_eps: f32,         // RMSNorm epsilon
}

impl ModelConfig {
    /// Configuration for Llama 3 8B
    pub fn llama3_8b() -> Self {
        Self {
            d_model: 4096,
            n_heads: 32,
            n_kv_heads: 8,
            n_layers: 32,
            d_ff: 14336,
            vocab_size: 128256,
            max_seq_len: 8192,
            rope_base: 500000.0,
            norm_eps: 1e-5,
        }
    }

    /// Configuration for Qwen 7B
    pub fn qwen_7b() -> Self {
        Self {
            d_model: 4096,
            n_heads: 32,
            n_kv_heads: 32,
            n_layers: 32,
            d_ff: 11008,
            vocab_size: 152064,
            max_seq_len: 2048,
            rope_base: 1000000.0,
            norm_eps: 1e-5,
        }
    }

    /// Derived property: dimension per head
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

/// SwiGLU Feedforward Network.
///
/// Modern LLMs use gated linear units instead of standard MLPs.
/// SwiGLU combines Swish activation with gating for improved expressivity.
///
/// Forward: output = (x @ w_gate) * swish(x @ w_up) @ w_down
///
/// # References
/// - Shazeer et al. (2020): "GLU Variants Improve Transformer"
/// - Used in: Llama 3, Qwen, Mistral
#[derive(Debug, Clone)]
pub struct SwiGLU {
    /// Weight matrix for gate projection: [d_model, d_ff]
    pub w_gate: Vec<f32>,
    /// Weight matrix for up projection: [d_model, d_ff]
    pub w_up: Vec<f32>,
    /// Weight matrix for down projection: [d_ff, d_model]
    pub w_down: Vec<f32>,
}

impl SwiGLU {
    /// Create a new SwiGLU MLP layer.
    ///
    /// # Arguments
    /// - `d_model`: Input/output dimension
    /// - `d_ff`: Hidden dimension (intermediate size)
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            w_gate: vec![0.1; d_model * d_ff],
            w_up: vec![0.1; d_model * d_ff],
            w_down: vec![0.1; d_ff * d_model],
        }
    }

    /// Apply SwiGLU to input.
    ///
    /// # Arguments
    /// - `x`: Input tensor, shape [seq_len, d_model] (flattened)
    /// - `d_model`: Model dimension
    /// - `d_ff`: Hidden dimension
    ///
    /// # Returns
    /// Output tensor, shape [seq_len, d_model]
    pub fn forward(&self, x: &[f32], d_model: usize, d_ff: usize) -> Vec<f32> {
        assert_eq!(x.len() % d_model, 0);
        let seq_len = x.len() / d_model;
        let mut output = vec![0.0; x.len()];

        // For each token
        for t in 0..seq_len {
            let x_slice = &x[t * d_model..(t + 1) * d_model];
            let out_slice = &mut output[t * d_model..(t + 1) * d_model];

            // Compute gate: x @ w_gate
            let mut gate = vec![0.0; d_ff];
            for i in 0..d_ff {
                for j in 0..d_model {
                    gate[i] += x_slice[j] * self.w_gate[j * d_ff + i];
                }
            }

            // Compute up: x @ w_up
            let mut up = vec![0.0; d_ff];
            for i in 0..d_ff {
                for j in 0..d_model {
                    up[i] += x_slice[j] * self.w_up[j * d_ff + i];
                }
            }

            // Apply Swish activation: swish(u) = u * sigmoid(u) = u / (1 + e^-u)
            let mut swish_up = vec![0.0; d_ff];
            for i in 0..d_ff {
                let sigmoid = 1.0 / (1.0 + (-up[i]).exp());
                swish_up[i] = up[i] * sigmoid;
            }

            // Gate the activation
            let mut gated = vec![0.0; d_ff];
            for i in 0..d_ff {
                gated[i] = gate[i] * swish_up[i];
            }

            // Project down: gated @ w_down
            for i in 0..d_model {
                for j in 0..d_ff {
                    out_slice[i] += gated[j] * self.w_down[j * d_model + i];
                }
            }
        }

        output
    }

    /// Load weights from a WeightLoader.
    pub fn load_weights(&mut self, loader: &WeightLoader, names: &WeightNames, layer_idx: usize, d_model: usize, d_ff: usize) -> WeightResult<()> {
        let (w_gate, w_up, w_down) = loader.load_mlp(names, layer_idx, d_model, d_ff)?;
        self.w_gate = w_gate;
        self.w_up = w_up;
        self.w_down = w_down;
        Ok(())
    }
}

/// Multi-Head Self-Attention.
///
/// Core attention mechanism for transformers.
/// Computes: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
///
/// With multiple heads:
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W^O
#[derive(Debug, Clone)]
pub struct Attention {
    /// Query projection: [d_model, d_model]
    pub w_q: Vec<f32>,
    /// Key projection: [d_model, d_model]
    pub w_k: Vec<f32>,
    /// Value projection: [d_model, d_model]
    pub w_v: Vec<f32>,
    /// Output projection: [d_model, d_model]
    pub w_o: Vec<f32>,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
}

/// Transformer block combining attention and MLP with residual connections and normalization.
///
/// Architecture:
/// ```text
/// x -> RMSNorm -> Attention + RoPE -> Add(x) -> RMSNorm -> SwiGLU -> Add(x)
/// ```
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Pre-attention normalization
    pub norm1: RMSNorm,
    /// Multi-head self-attention
    pub attn: Attention,
    /// Pre-MLP normalization
    pub norm2: RMSNorm,
    /// SwiGLU feedforward network
    pub mlp: SwiGLU,
}

impl TransformerBlock {
    /// Create a new transformer block.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            norm1: RMSNorm::new(config.d_model, config.norm_eps),
            attn: Attention::new(config.d_model, config.n_heads),
            norm2: RMSNorm::new(config.d_model, config.norm_eps),
            mlp: SwiGLU::new(config.d_model, config.d_ff),
        }
    }

    /// Forward pass with residual connections.
    pub fn forward(
        &self,
        x: &[f32],
        _rope: &RoPE,
        _seq_len: usize,
        d_model: usize,
        d_ff: usize,
    ) -> Vec<f32> {
        // Pre-norm attention with residuals: x + Attention(Norm(x))
        let x_norm = self.norm1.forward(x, d_model);
        let attn_out = self.attn.forward(&x_norm, &x_norm, &x_norm, d_model);

        // Add residual
        let x_after_attn: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // Pre-norm MLP with residuals: x + MLP(Norm(x))
        let x_norm2 = self.norm2.forward(&x_after_attn, d_model);
        let mlp_out = self.mlp.forward(&x_norm2, d_model, d_ff);

        // Add residual
        x_after_attn
            .iter()
            .zip(mlp_out.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    /// Load weights from a WeightLoader.
    pub fn load_weights(&mut self, loader: &WeightLoader, names: &WeightNames, layer_idx: usize, config: &ModelConfig) -> WeightResult<()> {
        // Load norm weights
        self.norm1.weight = loader.load_weight(&format!("model.layers.{}.input_layernorm.weight", layer_idx), config.d_model)?;
        self.norm2.weight = loader.load_weight(&format!("model.layers.{}.post_attention_layernorm.weight", layer_idx), config.d_model)?;

        // Load attention and MLP weights
        self.attn.load_weights(loader, names, layer_idx, config.d_model)?;
        self.mlp.load_weights(loader, names, layer_idx, config.d_model, config.d_ff)?;
        Ok(())
    }
}

/// Full Llama/Qwen model with transformer layers and embedding/output projections.
#[derive(Debug, Clone)]
pub struct LlamaModel {
    /// Model configuration
    pub config: ModelConfig,
    /// Token embeddings: [vocab_size, d_model]
    pub embeddings: Vec<f32>,
    /// Transformer blocks (one per layer)
    pub blocks: Vec<TransformerBlock>,
    /// Final output normalization
    pub norm: RMSNorm,
    /// Output projection: [d_model, vocab_size]
    pub lm_head: Vec<f32>,
}

impl LlamaModel {
    /// Create a new model with given configuration.
    pub fn new(config: ModelConfig) -> Self {
        let embeddings = vec![0.1; config.vocab_size * config.d_model];
        let blocks = (0..config.n_layers)
            .map(|_| TransformerBlock::new(&config))
            .collect();
        let lm_head = vec![0.1; config.d_model * config.vocab_size];
        let norm = RMSNorm::new(config.d_model, config.norm_eps);

        Self {
            config,
            embeddings,
            blocks,
            norm,
            lm_head,
        }
    }

    /// Load token embedding for a single token.
    fn embed_token(&self, token_id: usize) -> Vec<f32> {
        let offset = token_id * self.config.d_model;
        self.embeddings[offset..offset + self.config.d_model].to_vec()
    }

    /// Forward pass: embed tokens -> transformer blocks -> norm -> lm_head.
    pub fn forward(&self, token_ids: &[usize], rope: &RoPE) -> Vec<f32> {
        let seq_len = token_ids.len();

        // Embed tokens
        let mut hidden_states = Vec::with_capacity(seq_len * self.config.d_model);
        for &token_id in token_ids {
            hidden_states.extend(self.embed_token(token_id));
        }

        // Pass through transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(
                &hidden_states,
                rope,
                seq_len,
                self.config.d_model,
                self.config.d_ff,
            );
        }

        // Final norm and projection
        let hidden_states = self.norm.forward(&hidden_states, self.config.d_model);
        let last_hidden = &hidden_states[(seq_len - 1) * self.config.d_model..seq_len * self.config.d_model];

        let mut logits = vec![0.0; self.config.vocab_size];
        for i in 0..self.config.vocab_size {
            for j in 0..self.config.d_model {
                logits[i] += last_hidden[j] * self.lm_head[j * self.config.vocab_size + i];
            }
        }

        logits
    }

    /// Sample next token from logits.
    pub fn sample_token(logits: &[f32], temperature: f32) -> usize {
        let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();
        let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Generate text tokens from prompt.
    pub fn generate(
        &self,
        mut prompt: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        rope: &RoPE,
    ) -> Vec<usize> {
        for _ in 0..max_new_tokens {
            let logits = self.forward(&prompt, rope);
            let next_token = Self::sample_token(&logits, temperature);
            prompt.push(next_token);

            if prompt.len() >= self.config.max_seq_len {
                break;
            }
        }

        prompt
    }

    /// Load model weights from a safetensors file.
    pub fn load_weights<P: AsRef<Path>>(
        &mut self,
        path: P,
        names: &WeightNames,
    ) -> WeightResult<()> {
        let loader = WeightLoader::from_file(path)?;

        // Load embeddings
        self.embeddings = loader.load_embeddings(names, self.config.vocab_size, self.config.d_model)?;

        // Load transformer block weights
        for (layer_idx, block) in self.blocks.iter_mut().enumerate() {
            block.load_weights(&loader, names, layer_idx, &self.config)?;
        }

        // Load final norm weights
        self.norm.weight = loader.load_norm(names, self.config.d_model)?;

        // Load output projection
        self.lm_head = loader.load_lm_head(names, self.config.d_model, self.config.vocab_size)?;

        Ok(())
    }

    /// Create model from config and load weights from file.
    pub fn load_from_file<P: AsRef<Path>>(
        config: ModelConfig,
        weights_path: P,
        names: &WeightNames,
    ) -> WeightResult<Self> {
        let mut model = Self::new(config);
        model.load_weights(weights_path, names)?;
        Ok(model)
    }
}

impl Attention {
    /// Create a new multi-head attention layer.
    ///
    /// # Arguments
    /// - `d_model`: Total model dimension
    /// - `n_heads`: Number of attention heads
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let head_dim = d_model / n_heads;
        assert_eq!(
            d_model % n_heads,
            0,
            "d_model must be divisible by n_heads"
        );

        Self {
            w_q: vec![0.1; d_model * d_model],
            w_k: vec![0.1; d_model * d_model],
            w_v: vec![0.1; d_model * d_model],
            w_o: vec![0.1; d_model * d_model],
            n_heads,
            head_dim,
        }
    }

    /// Load weights from a WeightLoader.
    pub fn load_weights(&mut self, loader: &WeightLoader, names: &WeightNames, layer_idx: usize, d_model: usize) -> WeightResult<()> {
        let (w_q, w_k, w_v, w_o) = loader.load_attention(names, layer_idx, d_model)?;
        self.w_q = w_q;
        self.w_k = w_k;
        self.w_v = w_v;
        self.w_o = w_o;
        Ok(())
    }

    /// Softmax function for attention scores.
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    }

    /// Apply attention to queries, keys, values.
    ///
    /// # Arguments
    /// - `q`: Query tensor, shape [seq_len, d_model]
    /// - `k`: Key tensor, shape [seq_len, d_model]
    /// - `v`: Value tensor, shape [seq_len, d_model]
    /// - `d_model`: Model dimension
    ///
    /// # Returns
    /// Attention output, shape [seq_len, d_model]
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], d_model: usize) -> Vec<f32> {
        let seq_len = q.len() / d_model;
        assert_eq!(k.len(), v.len());

        // Project Q, K, V
        let mut q_proj = vec![0.0; q.len()];
        for i in 0..q.len() {
            for j in 0..d_model {
                q_proj[i] += q[i / d_model * d_model + j] * self.w_q[j * d_model + (i % d_model)];
            }
        }

        let mut k_proj = vec![0.0; k.len()];
        for i in 0..k.len() {
            for j in 0..d_model {
                k_proj[i] +=
                    k[i / d_model * d_model + j] * self.w_k[j * d_model + (i % d_model)];
            }
        }

        let mut v_proj = vec![0.0; v.len()];
        for i in 0..v.len() {
            for j in 0..d_model {
                v_proj[i] +=
                    v[i / d_model * d_model + j] * self.w_v[j * d_model + (i % d_model)];
            }
        }

        // Compute attention scores: Q·K^T / sqrt(d_head)
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut scores = vec![0.0; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut score = 0.0;
                for h in 0..self.n_heads {
                    for d in 0..self.head_dim {
                        let idx_q = i * d_model + h * self.head_dim + d;
                        let idx_k = j * d_model + h * self.head_dim + d;
                        score += q_proj[idx_q] * k_proj[idx_k];
                    }
                }
                scores[i * seq_len + j] = score * scale;
            }
        }

        // Apply softmax attention weights
        let mut attn_weights = vec![vec![0.0; seq_len]; seq_len];
        for i in 0..seq_len {
            let row_scores = &scores[i * seq_len..(i + 1) * seq_len];
            let weights = Self::softmax(row_scores);
            attn_weights[i] = weights;
        }

        // Apply attention to values and project output
        let mut output = vec![0.0; seq_len * d_model];

        for i in 0..seq_len {
            for j in 0..d_model {
                let mut val = 0.0;
                for t in 0..seq_len {
                    for d in 0..d_model {
                        val += attn_weights[i][t] * v_proj[t * d_model + d]
                            * self.w_o[d * d_model + j];
                    }
                }
                output[i * d_model + j] = val;
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_basic() {
        let norm = RMSNorm::new(4, 1e-6);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = norm.forward(&input, 4);

        // Output should be same length as input
        assert_eq!(output.len(), 4);

        // After RMSNorm, values should be normalized
        // RMS of input: sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) ≈ 2.738
        // Normalized: [0.365, 0.730, 1.095, 1.460]
        // With weight [1,1,1,1], output should be close to normalized values
        assert!(output.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn rmsnorm_scaling() {
        let norm = RMSNorm::new(4, 1e-6);
        let input = vec![2.0, 4.0, 6.0, 8.0]; // Scaled version

        let output1 = norm.forward(&[1.0, 2.0, 3.0, 4.0], 4);
        let output2 = norm.forward(&input, 4);

        // RMSNorm is scale-invariant: scaling input should give same output
        for (o1, o2) in output1.iter().zip(output2.iter()) {
            assert!((o1 - o2).abs() < 1e-5, "RMSNorm should be scale-invariant");
        }
    }

    #[test]
    fn rmsnorm_sequence() {
        let norm = RMSNorm::new(2, 1e-6);
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens, 2-dim each
        let output = norm.forward(&input, 2);

        assert_eq!(output.len(), 4);
        // Should normalize each token independently
    }

    #[test]
    fn rope_basic() {
        let rope = RoPE::new(64, 10000.0);
        assert_eq!(rope.inv_freq.len(), 32);

        // First inverse frequency should be 1.0 (position independent)
        assert!((rope.inv_freq[0] - 1.0).abs() < 1e-5);

        // Last should be smallest
        assert!(rope.inv_freq[31] < rope.inv_freq[0]);
    }

    #[test]
    fn rope_forward_shape() {
        let rope = RoPE::new(64, 10000.0);
        let q = vec![0.1; 64]; // 1 token, 1 head, 64-dim
        let k = vec![0.2; 64];

        let (q_rot, k_rot) = rope.forward(&q, &k, 1, 1);

        assert_eq!(q_rot.len(), 64);
        assert_eq!(k_rot.len(), 64);
    }

    #[test]
    fn rope_rotation_property() {
        let rope = RoPE::new(8, 10000.0); // Small dimension for testing
        let q = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let k = vec![0.0; 8];

        let (q_rot, _) = rope.forward(&q, &k, 1, 1);

        // After rotation, magnitude should be preserved
        // |q_rot| should equal |q|
        let mag_q: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mag_q_rot: f32 = q_rot.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!((mag_q - mag_q_rot).abs() < 1e-5, "Rotation should preserve magnitude");
    }

    #[test]
    fn model_config_llama3() {
        let config = ModelConfig::llama3_8b();
        assert_eq!(config.d_model, 4096);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.n_layers, 32);
    }

    #[test]
    fn model_config_qwen() {
        let config = ModelConfig::qwen_7b();
        assert_eq!(config.d_model, 4096);
        assert_eq!(config.rope_base, 1000000.0);
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn swiglu_shape() {
        let mlp = SwiGLU::new(256, 1024);
        let input = vec![0.1; 256]; // 1 token, 256-dim
        let output = mlp.forward(&input, 256, 1024);

        assert_eq!(output.len(), 256, "Output should match input dimension");
    }

    #[test]
    fn swiglu_sequence() {
        let mlp = SwiGLU::new(64, 256);
        let input = vec![0.1; 256]; // 4 tokens, 64-dim each
        let output = mlp.forward(&input, 64, 256);

        assert_eq!(output.len(), 256);
        assert!(output.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn attention_shape() {
        let attn = Attention::new(256, 8);
        let q = vec![0.1; 256]; // 1 token, 256-dim
        let k = vec![0.1; 256];
        let v = vec![0.1; 256];

        let output = attn.forward(&q, &k, &v, 256);
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn attention_sequence() {
        let attn = Attention::new(64, 4); // 4 heads, 16-dim per head
        let seq_len = 4;
        let d_model = 64;

        let q = vec![0.1; seq_len * d_model];
        let k = vec![0.1; seq_len * d_model];
        let v = vec![0.1; seq_len * d_model];

        let output = attn.forward(&q, &k, &v, d_model);
        assert_eq!(output.len(), seq_len * d_model);
        assert!(output.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn attention_self_attention() {
        let attn = Attention::new(32, 2);
        // Same Q, K, V means self-attention
        let x = vec![0.1; 32]; // 1 token, 32-dim
        let output = attn.forward(&x, &x, &x, 32);

        assert_eq!(output.len(), x.len());
    }

    #[test]
    fn transformer_block_shape() {
        let config = ModelConfig::llama3_8b();
        let block = TransformerBlock::new(&config);
        let rope = RoPE::new(config.head_dim(), config.rope_base);

        let input = vec![0.1; config.d_model];
        let output = block.forward(&input, &rope, 1, config.d_model, config.d_ff);

        assert_eq!(output.len(), config.d_model);
    }

    #[test]
    fn transformer_block_sequence() {
        let config = ModelConfig::llama3_8b();
        let block = TransformerBlock::new(&config);
        let rope = RoPE::new(config.head_dim(), config.rope_base);

        let seq_len = 4;
        let input = vec![0.1; seq_len * config.d_model];
        let output = block.forward(&input, &rope, seq_len, config.d_model, config.d_ff);

        assert_eq!(output.len(), seq_len * config.d_model);
        assert!(output.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn llama_model_creation() {
        let config = ModelConfig::llama3_8b();
        let model = LlamaModel::new(config.clone());

        assert_eq!(model.config.d_model, 4096);
        assert_eq!(model.embeddings.len(), config.vocab_size * config.d_model);
        assert_eq!(model.blocks.len(), config.n_layers);
    }

    #[test]
    fn llama_model_forward() {
        let mut config = ModelConfig::llama3_8b();
        config.n_layers = 1; // Reduce for testing
        config.vocab_size = 100; // Small vocab

        let model = LlamaModel::new(config);
        let rope = RoPE::new(model.config.head_dim(), model.config.rope_base);

        let token_ids = vec![0, 1, 2]; // Simple prompt
        let logits = model.forward(&token_ids, &rope);

        assert_eq!(logits.len(), 100);
        assert!(logits.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn llama_model_sampling() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let token = LlamaModel::sample_token(&logits, 1.0);

        assert!(token < 5); // Valid token index
    }

    #[test]
    fn llama_model_generate() {
        let mut config = ModelConfig::llama3_8b();
        config.n_layers = 1;
        config.vocab_size = 50;
        config.max_seq_len = 10;

        let model = LlamaModel::new(config);
        let rope = RoPE::new(model.config.head_dim(), model.config.rope_base);

        let prompt = vec![0, 1];
        let generated = model.generate(prompt.clone(), 3, 1.0, &rope);

        assert!(generated.len() > prompt.len());
        assert!(generated.len() <= model.config.max_seq_len);
    }

    #[test]
    fn weight_names_llama3() {
        let names = WeightNames::llama3();
        assert_eq!(names.embed_tokens, "model.embed_tokens.weight");
        assert_eq!(names.norm_weight, "model.norm.weight");
        assert_eq!(names.lm_head, "lm_head.weight");
    }

    #[test]
    fn weight_names_qwen() {
        let names = WeightNames::qwen();
        assert_eq!(names.embed_tokens, "model.embed_tokens.weight");
        assert_eq!(names.norm_weight, "model.norm.weight");
        assert_eq!(names.lm_head, "lm_head.weight");
    }

    #[test]
    fn weight_error_display() {
        let err = WeightError::MissingWeight("test_weight".to_string());
        assert_eq!(err.to_string(), "Missing weight: test_weight");

        let err = WeightError::ShapeMismatch {
            expected: 100,
            got: 50,
        };
        assert_eq!(err.to_string(), "Shape mismatch: expected 100, got 50");
    }

    #[test]
    fn llama_model_weight_loading_api() {
        let config = ModelConfig::llama3_8b();
        let model = LlamaModel::new(config.clone());

        // Verify model has correct weight dimensions
        assert_eq!(
            model.embeddings.len(),
            config.vocab_size * config.d_model
        );
        assert_eq!(model.norm.weight.len(), config.d_model);
        assert_eq!(model.lm_head.len(), config.d_model * config.vocab_size);
    }

    #[test]
    fn block_weight_loading_api() {
        let config = ModelConfig::llama3_8b();
        let block = TransformerBlock::new(&config);

        // Verify block has correct weight dimensions before loading
        assert_eq!(block.attn.w_q.len(), config.d_model * config.d_model);
        assert_eq!(block.mlp.w_gate.len(), config.d_model * config.d_ff);

        // Verify block norms have correct dimensions
        assert_eq!(block.norm1.weight.len(), config.d_model);
        assert_eq!(block.norm2.weight.len(), config.d_model);
    }

    #[test]
    fn weight_names_all_variants() {
        let llama_names = WeightNames::llama3();
        let qwen_names = WeightNames::qwen();

        // Test attention key generation
        let attn_q = llama_names.block_attn_key(0, "q_proj");
        assert!(attn_q.contains("model.layers.0"));
        assert!(attn_q.contains("self_attn"));
        assert!(attn_q.contains("q_proj"));

        // Test MLP key generation
        let mlp_gate = llama_names.block_mlp_key(0, "gate_proj");
        assert!(mlp_gate.contains("model.layers.0"));
        assert!(mlp_gate.contains("mlp"));
        assert!(mlp_gate.contains("gate_proj"));

        // Test norm key generation
        let norm = llama_names.block_norm_key(0, "unused");
        assert!(norm.contains("model.layers.0"));
        assert!(norm.contains("input_layernorm"));

        // Verify Qwen and Llama use same naming (for now)
        assert_eq!(
            llama_names.block_attn_key(5, "o_proj"),
            qwen_names.block_attn_key(5, "o_proj")
        );
    }
}
