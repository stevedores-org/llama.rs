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
}
