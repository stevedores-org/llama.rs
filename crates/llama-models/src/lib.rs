//! # llama-models
//!
//! Foundational model blocks for llama.rs:
//! - RMSNorm
//! - RoPE
//! - Attention (scaled dot-product, single-step decode form)
//! - MLP (SwiGLU)
//! - Safetensors weight loading

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use bytemuck::cast_slice;
use safetensors::{Dtype, SafeTensors};

/// Errors for model operations and weight loading.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("missing weight: {0}")]
    MissingWeight(String),
    #[error("invalid dtype for {name}: expected F32, got {dtype:?}")]
    InvalidDtype { name: String, dtype: Dtype },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
}

/// Lightweight tensor holder for loaded model weights.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Named weight storage loaded from safetensors.
#[derive(Debug, Clone, Default)]
pub struct ModelWeights {
    tensors: HashMap<String, Tensor>,
}

impl ModelWeights {
    pub fn load_safetensors_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        let st = SafeTensors::deserialize(bytes)?;
        let mut tensors = HashMap::new();

        for name in st.names() {
            let view = st.tensor(name)?;
            if view.dtype() != Dtype::F32 {
                return Err(ModelError::InvalidDtype {
                    name: name.to_string(),
                    dtype: view.dtype(),
                });
            }

            let shape = view.shape().to_vec();
            let raw: &[f32] = cast_slice(view.data());
            tensors.insert(
                name.to_string(),
                Tensor {
                    shape,
                    data: raw.to_vec(),
                },
            );
        }

        Ok(Self { tensors })
    }

    pub fn load_safetensors_file(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let bytes = fs::read(path)?;
        Self::load_safetensors_bytes(&bytes)
    }

    pub fn get(&self, name: &str) -> Result<&Tensor, ModelError> {
        self.tensors
            .get(name)
            .ok_or_else(|| ModelError::MissingWeight(name.to_string()))
    }
}

/// Root mean square normalization.
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, ModelError> {
    if input.len() != weight.len() {
        return Err(ModelError::Shape(format!(
            "rms_norm input/weight mismatch: {} != {}",
            input.len(),
            weight.len()
        )));
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    Ok(input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| x * inv_rms * w)
        .collect())
}

/// Apply rotary positional embeddings in-place to query and key vectors.
///
/// `q` and `k` are flattened `[n_heads * head_dim]`.
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    position: usize,
    n_heads: usize,
    head_dim: usize,
    base: f32,
) -> Result<(), ModelError> {
    let expected = n_heads * head_dim;
    if q.len() != expected || k.len() != expected {
        return Err(ModelError::Shape(format!(
            "rope q/k mismatch: expected {}, got q={}, k={}",
            expected,
            q.len(),
            k.len()
        )));
    }
    if !head_dim.is_multiple_of(2) {
        return Err(ModelError::Shape(format!(
            "head_dim must be even for RoPE, got {}",
            head_dim
        )));
    }

    for h in 0..n_heads {
        let offset = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let theta = (position as f32) / base.powf(i as f32 / head_dim as f32);
            let (sin_t, cos_t) = theta.sin_cos();

            let q0 = q[offset + i];
            let q1 = q[offset + i + 1];
            q[offset + i] = q0 * cos_t - q1 * sin_t;
            q[offset + i + 1] = q0 * sin_t + q1 * cos_t;

            let k0 = k[offset + i];
            let k1 = k[offset + i + 1];
            k[offset + i] = k0 * cos_t - k1 * sin_t;
            k[offset + i + 1] = k0 * sin_t + k1 * cos_t;
        }
    }
    Ok(())
}

/// Single-step decode attention:
/// - `query`: `[n_heads * head_dim]`
/// - `keys`/`values`: `[seq_len * n_heads * head_dim]`
/// - output: `[n_heads * head_dim]`
pub fn attention_decode(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, ModelError> {
    let q_expected = n_heads * head_dim;
    let kv_expected = seq_len * n_heads * head_dim;
    if query.len() != q_expected {
        return Err(ModelError::Shape(format!(
            "query shape mismatch: expected {}, got {}",
            q_expected,
            query.len()
        )));
    }
    if keys.len() != kv_expected || values.len() != kv_expected {
        return Err(ModelError::Shape(format!(
            "kv shape mismatch: expected {}, got k={}, v={}",
            kv_expected,
            keys.len(),
            values.len()
        )));
    }

    let mut out = vec![0.0; q_expected];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..n_heads {
        let qh = &query[h * head_dim..(h + 1) * head_dim];

        let mut scores = vec![0.0f32; seq_len];
        for (t, score) in scores.iter_mut().enumerate().take(seq_len) {
            let kh_off = (t * n_heads + h) * head_dim;
            let kh = &keys[kh_off..kh_off + head_dim];
            let dot = qh.iter().zip(kh.iter()).map(|(&a, &b)| a * b).sum::<f32>();
            *score = dot * scale;
        }

        let probs = softmax(&scores);
        for (t, &p) in probs.iter().enumerate() {
            let vh_off = (t * n_heads + h) * head_dim;
            let vh = &values[vh_off..vh_off + head_dim];
            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            for i in 0..head_dim {
                out_h[i] += p * vh[i];
            }
        }
    }

    Ok(out)
}

/// SwiGLU MLP:
/// - gate = silu(x * W_gate)
/// - up = x * W_up
/// - hidden = gate .* up
/// - out = hidden * W_down
pub fn mlp_swiglu(
    x: &[f32],
    w_gate: &[f32], // [d_model, d_ff]
    w_up: &[f32],   // [d_model, d_ff]
    w_down: &[f32], // [d_ff, d_model]
    d_model: usize,
    d_ff: usize,
) -> Result<Vec<f32>, ModelError> {
    if x.len() != d_model {
        return Err(ModelError::Shape(format!(
            "x shape mismatch: expected {}, got {}",
            d_model,
            x.len()
        )));
    }
    if w_gate.len() != d_model * d_ff || w_up.len() != d_model * d_ff {
        return Err(ModelError::Shape(
            "w_gate/w_up shape mismatch".to_string(),
        ));
    }
    if w_down.len() != d_ff * d_model {
        return Err(ModelError::Shape("w_down shape mismatch".to_string()));
    }

    let gate_pre = matvec_row_major(x, w_gate, d_model, d_ff);
    let up = matvec_row_major(x, w_up, d_model, d_ff);
    let hidden: Vec<f32> = gate_pre
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu(g) * u)
        .collect();
    Ok(matvec_row_major(&hidden, w_down, d_ff, d_model))
}

/// Minimal Llama block composition.
pub struct LlamaBlock;

impl LlamaBlock {
    pub fn forward(
        input: &[f32],
        norm_weight: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        d_model: usize,
        d_ff: usize,
    ) -> Result<Vec<f32>, ModelError> {
        let x = rms_norm(input, norm_weight, 1e-5)?;
        mlp_swiglu(&x, w_gate, w_up, w_down, d_model, d_ff)
    }
}

/// Minimal Qwen block composition (same block primitives at this stage).
pub struct QwenBlock;

impl QwenBlock {
    pub fn forward(
        input: &[f32],
        norm_weight: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        d_model: usize,
        d_ff: usize,
    ) -> Result<Vec<f32>, ModelError> {
        LlamaBlock::forward(input, norm_weight, w_gate, w_up, w_down, d_model, d_ff)
    }
}

fn matvec_row_major(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; out_dim];
    for i in 0..out_dim {
        let mut acc = 0.0f32;
        for j in 0..in_dim {
            acc += x[j] * w[j * out_dim + i];
        }
        out[i] = acc;
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use std::collections::BTreeMap;

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "idx={} actual={} expected={} tol={}",
                idx,
                a,
                e,
                tol
            );
        }
    }

    #[test]
    fn rms_norm_matches_reference() {
        // Reference values computed from numpy for eps=1e-5.
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        let y = rms_norm(&x, &w, 1e-5).unwrap();
        let expected = [0.36514813, 0.73029625, 1.0954444, 1.4605925];
        assert_close(&y, &expected, 1e-5);
    }

    #[test]
    fn rope_matches_reference() {
        let mut q = [1.0, 0.0, 0.0, 1.0];
        let mut k = q;
        apply_rope(&mut q, &mut k, 1, 1, 4, 10_000.0).unwrap();
        let expected = [0.5403023, 0.84147096, -0.009999833, 0.99995];
        assert_close(&q, &expected, 1e-5);
        assert_close(&k, &expected, 1e-5);
    }

    #[test]
    fn attention_matches_reference() {
        let query = [1.0, 0.0];
        let keys = [1.0, 0.0, 0.0, 1.0]; // seq=2, heads=1, dim=2
        let values = [10.0, 0.0, 0.0, 20.0];
        let out = attention_decode(&query, &keys, &values, 2, 1, 2).unwrap();
        let expected = [6.697615, 6.60477];
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn swiglu_and_blocks_produce_expected_shape() {
        let x = [0.5, -1.0];
        let norm = [1.0, 1.0];
        let w_gate = [1.0, 0.0, 0.0, 1.0];
        let w_up = [0.5, 0.0, 0.0, 0.5];
        let w_down = [1.0, 0.0, 0.0, 1.0];

        let y_llama = LlamaBlock::forward(&x, &norm, &w_gate, &w_up, &w_down, 2, 2).unwrap();
        let y_qwen = QwenBlock::forward(&x, &norm, &w_gate, &w_up, &w_down, 2, 2).unwrap();

        assert_eq!(y_llama.len(), 2);
        assert_eq!(y_qwen.len(), 2);
        assert_close(&y_llama, &y_qwen, 1e-6);
    }

    #[test]
    fn mlp_swiglu_matches_reference() {
        // Reference: x=[1.0, 2.0], d_model=2, d_ff=2
        // w_gate=[0.5, 0, 0, 0.5] (row-major) => gate_pre = [0.5, 1.0]
        // w_up=[1, 0, 0, 1] => up = [1.0, 2.0]
        // silu(0.5) = 0.5/(1+exp(-0.5)) ≈ 0.31123
        // silu(1.0) = 1.0/(1+exp(-1.0)) ≈ 0.73106
        // hidden = [0.31123*1.0, 0.73106*2.0] = [0.31123, 1.46212]
        // w_down=[1, 0, 0, 1] => out = [0.31123, 1.46212]
        let x = [1.0f32, 2.0];
        let w_gate = [0.5, 0.0, 0.0, 0.5];
        let w_up = [1.0, 0.0, 0.0, 1.0];
        let w_down = [1.0, 0.0, 0.0, 1.0];
        let out = mlp_swiglu(&x, &w_gate, &w_up, &w_down, 2, 2).unwrap();

        let silu_05 = 0.5 / (1.0 + (-0.5f32).exp());
        let silu_10 = 1.0 / (1.0 + (-1.0f32).exp());
        let expected = [silu_05 * 1.0, silu_10 * 2.0];
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    fn rms_norm_scale_invariance() {
        // RMSNorm is scale-invariant: norm(α·x) == norm(x) when weight=1
        let w = [1.0, 1.0, 1.0, 1.0];
        let x1 = [1.0, 2.0, 3.0, 4.0];
        let x2 = [2.0, 4.0, 6.0, 8.0]; // 2× scaled
        let y1 = rms_norm(&x1, &w, 1e-5).unwrap();
        let y2 = rms_norm(&x2, &w, 1e-5).unwrap();
        assert_close(&y1, &y2, 1e-5);
    }

    #[test]
    fn rope_preserves_magnitude() {
        // RoPE is a rotation, so |q_rot| == |q|
        let mut q = [1.0, 0.5, -0.3, 0.8];
        let mut k = [0.2, -0.1, 0.7, 0.4];
        let q_orig = q;
        let k_orig = k;
        apply_rope(&mut q, &mut k, 5, 1, 4, 10_000.0).unwrap();

        let mag_before: f32 = q_orig.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag_before - mag_after).abs() < 1e-5, "RoPE should preserve magnitude");

        let k_mag_before: f32 = k_orig.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_mag_after: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((k_mag_before - k_mag_after).abs() < 1e-5);
    }

    #[test]
    fn attention_single_token_is_identity_like() {
        // With seq_len=1, attention output = values (softmax of single score = 1.0)
        let query = [1.0, 0.0];
        let keys = [1.0, 0.0]; // seq=1, heads=1, dim=2
        let values = [3.0, 7.0];
        let out = attention_decode(&query, &keys, &values, 1, 1, 2).unwrap();
        assert_close(&out, &values, 1e-5);
    }

    #[test]
    fn loads_weights_from_safetensors() {
        let tensor = [1.0f32, 2.0, 3.0, 4.0];
        let view = TensorView::new(Dtype::F32, vec![2, 2], cast_slice(&tensor)).unwrap();
        let mut map = BTreeMap::new();
        map.insert("w".to_string(), view);
        let bytes = serialize(map, &None).unwrap();

        let weights = ModelWeights::load_safetensors_bytes(&bytes).unwrap();
        let w = weights.get("w").unwrap();
        assert_eq!(w.shape, vec![2, 2]);
        assert_eq!(w.data, tensor);
    }
}
