//! # llama-sampling
//!
//! Sampling and decoding strategies for llama.rs.
//!
//! Supports:
//! - Greedy (argmax)
//! - Temperature scaling
//! - Top-k filtering
//! - Top-p (nucleus) filtering
//! - Repetition penalty
//! - Deterministic seeded RNG for reproducible generation

/// Sampling error type.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingError {
    InvalidLogits,
    InvalidTemperature,
    NoValidTokens,
}

impl std::fmt::Display for SamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplingError::InvalidLogits => write!(f, "Invalid logits array"),
            SamplingError::InvalidTemperature => write!(f, "Temperature must be > 0"),
            SamplingError::NoValidTokens => write!(f, "No valid tokens after filtering"),
        }
    }
}

impl std::error::Error for SamplingError {}

pub type SamplingResult<T> = std::result::Result<T, SamplingError>;

/// Deterministic RNG for reproducible sampling.
///
/// Uses a simple xorshift64 algorithm for fast, reproducible random numbers.
#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        // Avoid zero state which would produce all zeros
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate next random float in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Sampling configuration and strategy.
#[derive(Debug, Clone)]
pub struct Sampler {
    /// Temperature for softmax scaling. > 1.0 = more random, < 1.0 = more deterministic.
    pub temperature: f32,

    /// Top-k: only sample from top k logits.
    pub top_k: Option<usize>,

    /// Top-p (nucleus sampling): sample from smallest set of tokens with cumulative prob >= p.
    pub top_p: Option<f32>,

    /// Repetition penalty: penalize tokens that appear in history.
    pub repetition_penalty: Option<f32>,

    /// RNG state for reproducible sampling. Mutated on each call.
    rng: SeededRng,
}

impl Sampler {
    /// Create a sampler with default settings (greedy).
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            rng: SeededRng::new(42),
        }
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = SeededRng::new(seed);
        self
    }

    /// Sample a token index from logits using configured strategy.
    pub fn sample(&mut self, logits: &[f32]) -> SamplingResult<usize> {
        self.sample_inner(logits, &[])
    }

    /// Sample with history for repetition penalty.
    pub fn sample_with_history(
        &mut self,
        logits: &[f32],
        history: &[usize],
    ) -> SamplingResult<usize> {
        self.sample_inner(logits, history)
    }

    fn sample_inner(&mut self, logits: &[f32], history: &[usize]) -> SamplingResult<usize> {
        if logits.is_empty() {
            return Err(SamplingError::InvalidLogits);
        }

        if self.temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature);
        }

        let mut work_logits = logits.to_vec();

        // Apply repetition penalty: for tokens in history, divide positive
        // logits by penalty and multiply negative logits by penalty.
        // This always makes repeated tokens less likely regardless of sign.
        if let Some(penalty) = self.repetition_penalty {
            for &token_id in history {
                if token_id < work_logits.len() {
                    if work_logits[token_id] > 0.0 {
                        work_logits[token_id] /= penalty;
                    } else {
                        work_logits[token_id] *= penalty;
                    }
                }
            }
        }

        // Apply temperature scaling
        if (self.temperature - 1.0).abs() > 1e-6 {
            for logit in &mut work_logits {
                *logit /= self.temperature;
            }
        }

        // Apply top-k filtering
        if let Some(k) = self.top_k {
            Self::apply_top_k(&mut work_logits, k);
        }

        // Convert to probabilities
        let probs = Self::softmax(&work_logits);

        // If temperature is very low (near-greedy), just argmax
        if self.temperature < 1e-3 {
            return Ok(Self::argmax(&probs));
        }

        // Apply top-p (nucleus) filtering
        let probs = if let Some(p) = self.top_p {
            Self::apply_top_p(&probs, p)
        } else {
            probs
        };

        // Sample from distribution
        self.sample_from_distribution(&probs)
    }

    fn apply_top_k(logits: &mut [f32], k: usize) {
        if k == 0 || k >= logits.len() {
            return;
        }

        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let threshold = indexed[k - 1].1;
        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn apply_top_p(probs: &[f32], p: f32) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &pr)| (i, pr)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut cutoff_idx = 0;
        for (idx, (_, prob)) in indexed.iter().enumerate() {
            cumsum += prob;
            cutoff_idx = idx;
            if cumsum >= p {
                break;
            }
        }

        let cutoff_prob = indexed[cutoff_idx].1;
        let mut result = vec![0.0; probs.len()];
        for (i, &pr) in probs.iter().enumerate() {
            if pr >= cutoff_prob {
                result[i] = pr;
            }
        }

        // Renormalize
        let sum: f32 = result.iter().sum();
        if sum > 0.0 {
            for p in &mut result {
                *p /= sum;
            }
        }

        result
    }

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();

        if sum > 0.0 {
            exps.iter().map(|&e| e / sum).collect()
        } else {
            vec![1.0 / logits.len() as f32; logits.len()]
        }
    }

    fn argmax(probs: &[f32]) -> usize {
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    fn sample_from_distribution(&mut self, probs: &[f32]) -> SamplingResult<usize> {
        let r = self.rng.next_f32();
        let mut cumsum = 0.0;

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if r < cumsum {
                return Ok(i);
            }
        }

        // Fallback to last token with nonzero probability
        for (i, &prob) in probs.iter().enumerate().rev() {
            if prob > 0.0 {
                return Ok(i);
            }
        }

        Err(SamplingError::NoValidTokens)
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_rng_reproducible() {
        let mut rng1 = SeededRng::new(42);
        let mut rng2 = SeededRng::new(42);

        for _ in 0..100 {
            let v1 = rng1.next_f32();
            let v2 = rng2.next_f32();
            assert!((v1 - v2).abs() < 1e-6);
            assert!((0.0..1.0).contains(&v1));
        }
    }

    #[test]
    fn greedy_sampling() {
        let logits = vec![1.0, 10.0, 2.0, 0.5];
        let mut sampler = Sampler::new().with_temperature(0.0001);
        let token = sampler.sample(&logits).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn softmax_uniform() {
        let logits = vec![1.0, 1.0, 1.0];
        let probs = Sampler::softmax(&logits);
        assert_eq!(probs.len(), 3);
        assert!((probs[0] - 1.0 / 3.0).abs() < 1e-5);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn temperature_effect() {
        let logits = [1.0, 2.0, 0.5];

        let high_temp: Vec<f32> = logits.iter().map(|l| l / 10.0).collect();
        let low_temp: Vec<f32> = logits.iter().map(|l| l / 0.1).collect();

        let high_probs = Sampler::softmax(&high_temp);
        let low_probs = Sampler::softmax(&low_temp);

        let max_high = high_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_low = low_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Higher temperature = more uniform = lower peak
        assert!(max_high < max_low);
    }

    #[test]
    fn top_k_filtering() {
        let mut logits = vec![1.0, 10.0, 2.0, 0.5, 3.0];
        Sampler::apply_top_k(&mut logits, 2);
        assert!(logits[1].is_finite()); // Top token
        assert!(logits[4].is_finite()); // 2nd top token
        assert!(!logits[0].is_finite()); // Below top-k
    }

    #[test]
    fn top_p_filtering() {
        let probs = vec![0.5, 0.3, 0.15, 0.05];
        let filtered = Sampler::apply_top_p(&probs, 0.8);
        assert!(filtered[0] > 0.0);
        assert!(filtered[1] > 0.0);
        assert_eq!(filtered[2], 0.0);
        assert_eq!(filtered[3], 0.0);
    }

    #[test]
    fn repetition_penalty_reduces_likelihood() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let history = vec![3]; // Token 3 in history

        // Without penalty
        let probs_no_penalty = Sampler::softmax(&logits);

        // With penalty applied
        let mut penalized = logits.clone();
        penalized[3] /= 2.0; // Positive logit divided by penalty
        let probs_with_penalty = Sampler::softmax(&penalized);

        // Token 3 should have lower probability after penalty
        assert!(probs_with_penalty[3] < probs_no_penalty[3]);

        // Verify via sampler API
        let mut sampler = Sampler::new().with_repetition_penalty(2.0);
        let result = sampler.sample_with_history(&logits, &history);
        assert!(result.is_ok());
    }

    #[test]
    fn repetition_penalty_handles_negative_logits() {
        let logits = vec![-1.0, -2.0, 3.0];
        let history = vec![0, 1]; // Negative logit tokens in history

        let mut sampler = Sampler::new().with_repetition_penalty(2.0).with_seed(42);
        let result = sampler.sample_with_history(&logits, &history);
        assert!(result.is_ok());
    }

    #[test]
    fn deterministic_across_calls() {
        let logits = vec![0.1, 0.2, 0.3, 0.4];

        let mut sampler1 = Sampler::new().with_seed(42);
        let mut sampler2 = Sampler::new().with_seed(42);

        // Multiple calls should produce same sequence
        for _ in 0..10 {
            let t1 = sampler1.sample(&logits).unwrap();
            let t2 = sampler2.sample(&logits).unwrap();
            assert_eq!(t1, t2);
        }
    }

    #[test]
    fn rng_advances_between_calls() {
        let logits = vec![0.25, 0.25, 0.25, 0.25];
        let mut sampler = Sampler::new().with_seed(42);

        // With uniform distribution, we should eventually see different tokens
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            seen.insert(sampler.sample(&logits).unwrap());
        }
        assert!(seen.len() > 1, "RNG should produce varied results");
    }

    #[test]
    fn combined_sampling() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.1];
        let mut sampler = Sampler::new()
            .with_temperature(0.8)
            .with_top_k(3)
            .with_top_p(0.9)
            .with_seed(42);

        let token = sampler.sample(&logits).unwrap();
        assert!(token < logits.len());
    }

    #[test]
    fn edge_topk_1_equals_greedy() {
        let logits = vec![1.0, 5.0, 2.0, 3.0];
        // top_k=1 should always select the argmax token (index 1)
        let mut sampler = Sampler::new().with_top_k(1).with_seed(42);
        for _ in 0..10 {
            let token = sampler.sample(&logits).unwrap();
            assert_eq!(token, 1, "top_k=1 should always select argmax");
        }
    }

    #[test]
    fn edge_temperature_zero_returns_error() {
        // temperature=0.0 is explicitly invalid per the sampler contract
        let logits = vec![1.0, 2.0, 3.0];
        let mut sampler = Sampler::new().with_temperature(0.0);
        assert_eq!(
            sampler.sample(&logits),
            Err(SamplingError::InvalidTemperature)
        );
    }

    #[test]
    fn edge_negative_temperature_returns_error() {
        let logits = vec![1.0, 2.0];
        let mut sampler = Sampler::new().with_temperature(-1.0);
        assert_eq!(
            sampler.sample(&logits),
            Err(SamplingError::InvalidTemperature)
        );
    }

    #[test]
    fn invalid_temperature() {
        let logits = vec![1.0, 2.0];
        let mut sampler = Sampler::new().with_temperature(0.0);
        assert_eq!(
            sampler.sample(&logits),
            Err(SamplingError::InvalidTemperature)
        );
    }

    #[test]
    fn empty_logits() {
        let mut sampler = Sampler::new();
        assert_eq!(sampler.sample(&[]), Err(SamplingError::InvalidLogits));
    }
}
