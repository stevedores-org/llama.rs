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
//!
//! ## Example
//!
//! ```ignore
//! let mut sampler = Sampler::new()
//!     .with_temperature(0.8)
//!     .with_top_p(0.9);
//!
//! let token = sampler.sample(&logits)?;
//! ```

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

/// Deterministic RNG for reproducible sampling (for testing).
#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    /// Create RNG with a given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next random float in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // Simple LCG: state = (state * 1103515245 + 12345) % 2^31
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state >> 16) & 0x7fff) as f32 / 32768.0
    }

    /// Generate next random float in [0, 1) with same distribution.
    pub fn next_uniform(&mut self) -> f32 {
        self.next_f32()
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

    /// Minimum probability threshold for valid tokens.
    pub min_prob: f32,

    /// RNG seed for reproducible sampling.
    pub seed: u64,
}

impl Sampler {
    /// Create a sampler with default settings (greedy).
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            min_prob: 1e-6,
            seed: 42,
        }
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set top-k filtering.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Set top-p (nucleus) filtering.
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set repetition penalty.
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }

    /// Set RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sample a token from logits using configured strategy.
    pub fn sample(&self, logits: &[f32]) -> SamplingResult<usize> {
        if logits.is_empty() {
            return Err(SamplingError::InvalidLogits);
        }

        if self.temperature <= 0.0 {
            return Err(SamplingError::InvalidTemperature);
        }

        // Make a copy to work with
        let mut work_logits = logits.to_vec();

        // Apply temperature scaling
        if (self.temperature - 1.0).abs() > 1e-6 {
            for logit in &mut work_logits {
                *logit /= self.temperature;
            }
        }

        // Apply top-k filtering
        if let Some(k) = self.top_k {
            work_logits = self.apply_top_k(&work_logits, k);
        }

        // Convert to probabilities
        let mut probs = self.softmax(&work_logits);

        // Apply top-p (nucleus) filtering
        if let Some(p) = self.top_p {
            probs = self.apply_top_p(&probs, p);
        }

        // If temperature is very low (greedy)
        if self.temperature < 1e-3 {
            return Ok(self.argmax(&probs));
        }

        // Sample from distribution
        let mut rng = SeededRng::new(self.seed);
        self.sample_from_distribution(&probs, &mut rng)
    }

    /// Sample with history for repetition penalty.
    pub fn sample_with_history(&self, logits: &[f32], history: &[usize]) -> SamplingResult<usize> {
        if logits.is_empty() {
            return Err(SamplingError::InvalidLogits);
        }

        let mut work_logits = logits.to_vec();

        // Apply repetition penalty
        if let Some(penalty) = self.repetition_penalty {
            for &token_id in history {
                if token_id < work_logits.len() {
                    work_logits[token_id] /= penalty;
                }
            }
        }

        // Use same sampling as without history
        let saved_seed = self.seed;
        let sampler = Sampler {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: None, // Already applied
            min_prob: self.min_prob,
            seed: saved_seed,
        };

        sampler.sample(&work_logits)
    }

    fn apply_top_k(&self, logits: &[f32], k: usize) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = k.min(logits.len());
        let threshold = indexed[k - 1].1;

        let mut result = vec![f32::NEG_INFINITY; logits.len()];
        for (i, &logit) in logits.iter().enumerate() {
            if logit >= threshold {
                result[i] = logit;
            }
        }

        result
    }

    fn apply_top_p(&self, probs: &[f32], p: f32) -> Vec<f32> {
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

    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();

        let sum: f32 = exps.iter().sum();

        if sum > 0.0 {
            exps.iter().map(|&e| e / sum).collect()
        } else {
            vec![1.0 / logits.len() as f32; logits.len()]
        }
    }

    fn argmax(&self, probs: &[f32]) -> usize {
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    fn sample_from_distribution(
        &self,
        probs: &[f32],
        rng: &mut SeededRng,
    ) -> SamplingResult<usize> {
        let r = rng.next_uniform();
        let mut cumsum = 0.0;

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if r <= cumsum || cumsum >= 1.0 - self.min_prob {
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

        for _ in 0..10 {
            let v1 = rng1.next_f32();
            let v2 = rng2.next_f32();
            assert!((v1 - v2).abs() < 1e-6);
        }
    }

    #[test]
    fn greedy_sampling() {
        let logits = vec![1.0, 10.0, 2.0, 0.5];
        let sampler = Sampler::new().with_temperature(0.001); // Near-greedy
        let token = sampler.sample(&logits).unwrap();
        assert_eq!(token, 1); // Highest logit
    }

    #[test]
    fn softmax_calculation() {
        let sampler = Sampler::new();
        let logits = vec![1.0, 1.0, 1.0];
        let probs = sampler.softmax(&logits);

        assert_eq!(probs.len(), 3);
        assert!((probs[0] - 1.0 / 3.0).abs() < 1e-5);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn temperature_scaling() {
        let logits = vec![1.0, 2.0, 0.5];

        // High temperature = more uniform
        let high_temp_logits: Vec<f32> = logits.iter().map(|l| l / 10.0).collect();
        let high_temp = Sampler::new();
        let high_probs = high_temp.softmax(&high_temp_logits);

        // Low temperature = more peaked
        let low_temp_logits: Vec<f32> = logits.iter().map(|l| l / 0.1).collect();
        let low_temp = Sampler::new();
        let low_probs = low_temp.softmax(&low_temp_logits);

        // High temp should be more uniform (all probs closer to mean)
        let max_high = *high_probs
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_low = *low_probs
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        assert!(max_high < max_low); // Peak is less pronounced with high temp
    }

    #[test]
    fn top_k_filtering() {
        let logits = vec![1.0, 10.0, 2.0, 0.5, 3.0];
        let sampler = Sampler::new().with_top_k(2);

        let filtered = sampler.apply_top_k(&logits, 2);
        assert!(filtered[1].is_finite()); // Top token
        assert!(filtered[4].is_finite()); // 2nd top token
        assert!(!filtered[0].is_finite()); // Below top-k
    }

    #[test]
    fn top_p_filtering() {
        let probs = vec![0.5, 0.3, 0.15, 0.05];
        let sampler = Sampler::new();

        let filtered = sampler.apply_top_p(&probs, 0.8);
        assert!(filtered[0] > 0.0); // High prob
        assert!(filtered[1] > 0.0); // High prob
        assert_eq!(filtered[2], 0.0); // Below threshold
        assert_eq!(filtered[3], 0.0); // Below threshold
    }

    #[test]
    fn repetition_penalty() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let history = vec![3]; // Token 3 was sampled recently

        let sampler = Sampler::new().with_repetition_penalty(2.0);
        let result = sampler.sample_with_history(&logits, &history);

        assert!(result.is_ok());
        // Token 3's logit was penalized (4.0 / 2.0 = 2.0), so it's less likely
    }

    #[test]
    fn deterministic_generation() {
        let logits = vec![0.1, 0.2, 0.3, 0.4];

        let sampler1 = Sampler::new().with_seed(42);
        let token1 = sampler1.sample(&logits).unwrap();

        let sampler2 = Sampler::new().with_seed(42);
        let token2 = sampler2.sample(&logits).unwrap();

        assert_eq!(token1, token2);
    }

    #[test]
    fn combined_sampling() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.1];

        let sampler = Sampler::new()
            .with_temperature(0.8)
            .with_top_k(3)
            .with_top_p(0.9)
            .with_seed(42);

        let result = sampler.sample(&logits);
        assert!(result.is_ok());
        let token = result.unwrap();
        assert!(token < logits.len());
    }

    #[test]
    fn invalid_temperature() {
        let logits = vec![1.0, 2.0];
        let sampler = Sampler::new().with_temperature(0.0);

        let result = sampler.sample(&logits);
        assert_eq!(result, Err(SamplingError::InvalidTemperature));
    }

    #[test]
    fn empty_logits() {
        let logits = vec![];
        let sampler = Sampler::new();

        let result = sampler.sample(&logits);
        assert_eq!(result, Err(SamplingError::InvalidLogits));
    }
}
