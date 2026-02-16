//! # llama-sampling
//!
//! Sampling and decoding strategies for llama.rs.
//! Implements greedy, temperature, top-k, top-p, and repetition penalty sampling
//! with deterministic seeded RNG for reproducible test runs.

use std::borrow::Cow;
use std::cmp::Ordering;

/// Error type for sampling operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SamplingError {
    #[error("logits cannot be empty")]
    EmptyLogits,
    #[error("invalid token id in history: {0}")]
    InvalidHistoryToken(i32),
    #[error("temperature must be > 0, got {0}")]
    InvalidTemperature(f32),
    #[error("top_p must be in (0, 1], got {0}")]
    InvalidTopP(f32),
    #[error("repetition_penalty must be >= 1.0, got {0}")]
    InvalidRepetitionPenalty(f32),
}

/// Sampling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Pick argmax directly.
    Greedy,
    /// Sample from filtered probability distribution.
    Stochastic,
}

/// Configuration for token sampling.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub strategy: SamplingStrategy,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::Stochastic,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            seed: 0,
        }
    }
}

impl SamplingConfig {
    fn validate(&self) -> Result<(), SamplingError> {
        if self.temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature(self.temperature));
        }
        if let Some(top_p) = self.top_p {
            if !(top_p > 0.0 && top_p <= 1.0) {
                return Err(SamplingError::InvalidTopP(top_p));
            }
        }
        if let Some(penalty) = self.repetition_penalty {
            if penalty < 1.0 {
                return Err(SamplingError::InvalidRepetitionPenalty(penalty));
            }
        }
        Ok(())
    }
}

/// Stateful sampler using deterministic RNG.
pub struct Sampler {
    cfg: SamplingConfig,
    rng: XorShift64,
}

impl Sampler {
    pub fn new(cfg: SamplingConfig) -> Result<Self, SamplingError> {
        cfg.validate()?;
        Ok(Self {
            rng: XorShift64::seeded(cfg.seed),
            cfg,
        })
    }

    /// Sample a token from logits with optional repetition penalty against `history`.
    pub fn sample(&mut self, logits: &[f32], history: &[i32]) -> Result<i32, SamplingError> {
        if logits.is_empty() {
            return Err(SamplingError::EmptyLogits);
        }

        let adjusted: Cow<'_, [f32]> = if let Some(penalty) = self.cfg.repetition_penalty {
            let mut buf = logits.to_vec();
            apply_repetition_penalty(&mut buf, history, penalty)?;
            Cow::Owned(buf)
        } else {
            Cow::Borrowed(logits)
        };

        if self.cfg.strategy == SamplingStrategy::Greedy {
            return greedy_sample(&adjusted);
        }

        let mut probs = softmax_with_temperature(&adjusted, self.cfg.temperature)?;

        if let Some(top_k) = self.cfg.top_k {
            apply_top_k(&mut probs, top_k);
            normalize_probs(&mut probs);
        }

        if let Some(top_p) = self.cfg.top_p {
            apply_top_p(&mut probs, top_p)?;
        }

        normalize_probs(&mut probs);
        Ok(sample_from_probs(&probs, &mut self.rng))
    }
}

/// Greedy decoding (argmax).
pub fn greedy_sample(logits: &[f32]) -> Result<i32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }

    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (idx, &val) in logits.iter().enumerate().skip(1) {
        if val > best_val {
            best_idx = idx;
            best_val = val;
        }
    }
    Ok(best_idx as i32)
}

/// Apply repetition penalty to logits for tokens present in `history`.
///
/// Rule follows common practice:
/// - positive logit: divide by penalty
/// - negative logit: multiply by penalty
pub fn apply_repetition_penalty(
    logits: &mut [f32],
    history: &[i32],
    penalty: f32,
) -> Result<(), SamplingError> {
    if penalty < 1.0 {
        return Err(SamplingError::InvalidRepetitionPenalty(penalty));
    }

    let mut seen = vec![false; logits.len()];
    for &token in history {
        if token < 0 {
            return Err(SamplingError::InvalidHistoryToken(token));
        }
        let idx = token as usize;
        if idx >= logits.len() {
            return Err(SamplingError::InvalidHistoryToken(token));
        }
        if !seen[idx] {
            seen[idx] = true;
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
    Ok(())
}

fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Result<Vec<f32>, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature(temperature));
    }

    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if max_val == f32::NEG_INFINITY {
        // All logits are -inf, return uniform distribution
        return Ok(vec![1.0 / logits.len() as f32; logits.len()]);
    }

    let mut exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    normalize_probs(&mut exps);
    Ok(exps)
}

fn normalize_probs(probs: &mut [f32]) {
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }
}

fn apply_top_k(probs: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= probs.len() {
        return;
    }

    let mut order: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    order.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    for &(idx, _) in order.iter().skip(top_k) {
        probs[idx] = 0.0;
    }
}

fn apply_top_p(probs: &mut [f32], top_p: f32) -> Result<(), SamplingError> {
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(SamplingError::InvalidTopP(top_p));
    }

    let mut order: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    order.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut cumulative = 0.0f32;
    let mut keep = vec![false; probs.len()];
    for &(idx, p) in &order {
        cumulative += p;
        keep[idx] = true;
        if cumulative >= top_p {
            break;
        }
    }

    for (idx, p) in probs.iter_mut().enumerate() {
        if !keep[idx] {
            *p = 0.0;
        }
    }
    Ok(())
}

fn sample_from_probs(probs: &[f32], rng: &mut XorShift64) -> i32 {
    let r = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (idx, &p) in probs.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        cumulative += p;
        if r < cumulative {
            return idx as i32;
        }
    }

    // If all probs are zero after filtering, fall back to argmax.
    probs
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.0.cmp(&a.0))
        })
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn seeded(seed: u64) -> Self {
        // Avoid stuck zero state.
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        let v = self.next_u64() >> 40; // 24 bits
        (v as f32) / ((1u32 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_selects_max_logit() {
        let logits = vec![0.1, 2.0, 1.5];
        assert_eq!(greedy_sample(&logits).unwrap(), 1);
    }

    #[test]
    fn top_k_limits_candidates() {
        let cfg = SamplingConfig {
            top_k: Some(1),
            seed: 42,
            ..SamplingConfig::default()
        };
        let mut sampler = Sampler::new(cfg).unwrap();
        let logits = vec![0.1, 5.0, 4.0];
        assert_eq!(sampler.sample(&logits, &[]).unwrap(), 1);
    }

    #[test]
    fn top_p_limits_tail() {
        let cfg = SamplingConfig {
            top_p: Some(0.55),
            seed: 42,
            ..SamplingConfig::default()
        };
        let mut sampler = Sampler::new(cfg).unwrap();
        let logits = vec![4.0, 2.0, 1.0];
        // top_p should keep only token 0 for this distribution.
        assert_eq!(sampler.sample(&logits, &[]).unwrap(), 0);
    }

    #[test]
    fn seeded_rng_is_deterministic() {
        let cfg = SamplingConfig {
            top_k: Some(3),
            top_p: Some(0.95),
            temperature: 0.9,
            seed: 12345,
            ..SamplingConfig::default()
        };
        let mut a = Sampler::new(cfg.clone()).unwrap();
        let mut b = Sampler::new(cfg).unwrap();

        let logits = vec![1.0, 1.1, 1.2, 1.3];
        let mut seq_a = Vec::new();
        let mut seq_b = Vec::new();
        for _ in 0..20 {
            seq_a.push(a.sample(&logits, &seq_a).unwrap());
            seq_b.push(b.sample(&logits, &seq_b).unwrap());
        }

        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn repetition_penalty_masks_logits() {
        let cfg = SamplingConfig {
            strategy: SamplingStrategy::Greedy,
            repetition_penalty: Some(2.0),
            ..SamplingConfig::default()
        };
        let mut sampler = Sampler::new(cfg).unwrap();

        let logits = vec![0.9, 1.0];
        let history = vec![1];
        // token 1 becomes 0.5 after penalty, so token 0 should win.
        assert_eq!(sampler.sample(&logits, &history).unwrap(), 0);
    }

    #[test]
    fn softmax_handles_all_inf() {
        let logits = vec![f32::NEG_INFINITY, f32::NEG_INFINITY];
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        assert_eq!(probs.len(), 2);
        assert_eq!(probs[0], 0.5);
        assert_eq!(probs[1], 0.5);
    }

    #[test]
    fn invalid_config_is_rejected() {
        let cfg = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        assert!(matches!(
            Sampler::new(cfg),
            Err(SamplingError::InvalidTemperature(0.0))
        ));
    }
}
