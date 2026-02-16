//! # llama-runtime
//!
//! Runtime execution and verification helpers for llama.rs.
//!
//! This crate includes a Phase-1 verification harness for LLAMA-006:
//! `full_forward(prompt)` logits vs `prefill(prompt[:-1]) + decode(last_token)` logits.

use llama_kv::{KVLayout, LayerKVCache};

/// Errors for runtime verification routines.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("prompt must contain at least 2 tokens")]
    PromptTooShort,
    #[error("invalid token id: {0}")]
    InvalidToken(i32),
    #[error("kv error: {0}")]
    Kv(#[from] llama_kv::KVError),
}

/// Result of a KV equivalence run.
#[derive(Debug, Clone, Copy)]
pub struct KvEquivalenceReport {
    pub max_abs_diff: f32,
}

/// Minimal deterministic runtime verifier used for LLAMA-006 true tests.
pub struct RuntimeVerifier {
    model: ToyModel,
}

impl Default for RuntimeVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeVerifier {
    pub fn new() -> Self {
        Self {
            model: ToyModel::default(),
        }
    }

    /// True test:
    /// full_forward(prompt) logits == prefill(prompt[:-1]) + decode(last_token) logits.
    pub fn verify_kv_equivalence(
        &self,
        prompt: &[i32],
    ) -> Result<KvEquivalenceReport, RuntimeError> {
        if prompt.len() < 2 {
            return Err(RuntimeError::PromptTooShort);
        }

        let full = self.model.full_forward(prompt)?;
        let kv = self.model.prefill_then_decode(prompt, false)?;
        Ok(KvEquivalenceReport {
            max_abs_diff: max_abs_diff(&full, &kv),
        })
    }

    /// Same flow as `verify_kv_equivalence` but injects an off-by-one position bug
    /// in decode. Used to prove the true test catches indexing errors.
    pub fn verify_with_off_by_one_bug(
        &self,
        prompt: &[i32],
    ) -> Result<KvEquivalenceReport, RuntimeError> {
        if prompt.len() < 2 {
            return Err(RuntimeError::PromptTooShort);
        }

        let full = self.model.full_forward(prompt)?;
        let kv_bug = self.model.prefill_then_decode(prompt, true)?;
        Ok(KvEquivalenceReport {
            max_abs_diff: max_abs_diff(&full, &kv_bug),
        })
    }
}

#[derive(Debug)]
struct ToyModel {
    embeddings: [[f32; 2]; 8],
    out_proj: [[f32; 8]; 2],
}

impl Default for ToyModel {
    fn default() -> Self {
        Self {
            embeddings: [
                [0.4, -0.2],
                [0.1, 0.9],
                [0.8, 0.2],
                [-0.5, 0.7],
                [0.3, -0.9],
                [-0.2, -0.3],
                [0.6, 0.4],
                [-0.7, 0.5],
            ],
            out_proj: [
                [0.3, -0.1, 0.2, 0.5, -0.4, 0.1, 0.2, -0.3],
                [-0.2, 0.6, -0.3, 0.1, 0.4, -0.5, 0.2, 0.3],
            ],
        }
    }
}

impl ToyModel {
    fn full_forward(&self, prompt: &[i32]) -> Result<Vec<f32>, RuntimeError> {
        let seq_len = prompt.len();
        let mut keys = Vec::with_capacity(seq_len * 2);
        let mut values = Vec::with_capacity(seq_len * 2);

        for (pos, &tok) in prompt.iter().enumerate() {
            let mut kv = self.token_vec(tok)?;
            apply_position_rotation(&mut kv, pos);
            keys.extend_from_slice(&kv);
            values.extend_from_slice(&kv);
        }

        let mut q = self.token_vec(*prompt.last().expect("prompt checked non-empty"))?;
        apply_position_rotation(&mut q, seq_len - 1);

        let ctx = attention_single_head(&q, &keys, &values, seq_len, 2);
        Ok(project_logits(&ctx, &self.out_proj))
    }

    fn prefill_then_decode(
        &self,
        prompt: &[i32],
        inject_off_by_one: bool,
    ) -> Result<Vec<f32>, RuntimeError> {
        let prefill_len = prompt.len() - 1;
        let mut cache = LayerKVCache::new(prompt.len(), 1, 2, KVLayout::BySequence);

        let mut k_prefill = Vec::with_capacity(prefill_len * 2);
        let mut v_prefill = Vec::with_capacity(prefill_len * 2);
        for (pos, &tok) in prompt[..prefill_len].iter().enumerate() {
            let mut kv = self.token_vec(tok)?;
            apply_position_rotation(&mut kv, pos);
            k_prefill.extend_from_slice(&kv);
            v_prefill.extend_from_slice(&kv);
        }
        cache.write_prefill(&k_prefill, &v_prefill, prefill_len)?;

        let last = *prompt.last().expect("prompt checked len >= 2");
        let mut q = self.token_vec(last)?;
        let decode_pos = if inject_off_by_one {
            prefill_len + 1
        } else {
            prefill_len
        };
        apply_position_rotation(&mut q, decode_pos);

        let mut kv_last = self.token_vec(last)?;
        apply_position_rotation(&mut kv_last, decode_pos);
        cache.append_token(&kv_last, &kv_last)?;

        let seq_len = cache.seq_len;
        let keys = cache.k[..seq_len * 2].to_vec();
        let values = cache.v[..seq_len * 2].to_vec();
        let ctx = attention_single_head(&q, &keys, &values, seq_len, 2);
        Ok(project_logits(&ctx, &self.out_proj))
    }

    fn token_vec(&self, token: i32) -> Result<[f32; 2], RuntimeError> {
        let idx = usize::try_from(token).map_err(|_| RuntimeError::InvalidToken(token))?;
        self.embeddings
            .get(idx)
            .copied()
            .ok_or(RuntimeError::InvalidToken(token))
    }
}

fn apply_position_rotation(v: &mut [f32; 2], position: usize) {
    let theta = position as f32 * 0.15;
    let (sin_t, cos_t) = theta.sin_cos();
    let x0 = v[0];
    let x1 = v[1];
    v[0] = x0 * cos_t - x1 * sin_t;
    v[1] = x0 * sin_t + x1 * cos_t;
}

fn attention_single_head(
    q: &[f32; 2],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    dim: usize,
) -> [f32; 2] {
    let scale = 1.0 / (dim as f32).sqrt();

    let mut scores = vec![0.0f32; seq_len];
    for t in 0..seq_len {
        let k = &keys[t * dim..t * dim + dim];
        scores[t] = (q[0] * k[0] + q[1] * k[1]) * scale;
    }
    let probs = softmax(&scores);

    let mut out = [0.0f32; 2];
    for (t, &p) in probs.iter().enumerate() {
        let v = &values[t * dim..t * dim + dim];
        out[0] += p * v[0];
        out[1] += p * v[1];
    }
    out
}

fn project_logits(ctx: &[f32; 2], out_proj: &[[f32; 8]; 2]) -> Vec<f32> {
    let mut logits = vec![0.0f32; 8];
    for (i, logit) in logits.iter_mut().enumerate().take(8) {
        *logit = ctx[0] * out_proj[0][i] + ctx[1] * out_proj[1][i];
    }
    logits
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max_v = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = scores.iter().map(|s| (s - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_true_test_equivalence_holds() {
        let verifier = RuntimeVerifier::new();
        let prompt = [1, 3, 2, 6];
        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "expected <= 1e-5, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_true_test_detects_off_by_one_bug() {
        let verifier = RuntimeVerifier::new();
        let prompt = [1, 3, 2, 6];
        let report = verifier.verify_with_off_by_one_bug(&prompt).unwrap();
        assert!(
            report.max_abs_diff > 1e-4,
            "off-by-one bug should be detectable, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn verify_rejects_too_short_prompt() {
        let verifier = RuntimeVerifier::new();
        let err = verifier.verify_kv_equivalence(&[1]).unwrap_err();
        assert!(matches!(err, RuntimeError::PromptTooShort));
    }
}
