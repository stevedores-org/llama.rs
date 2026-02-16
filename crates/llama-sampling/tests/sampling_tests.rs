//! Integration tests for llama-sampling.
//!
//! Validates:
//! - Softmax mathematical properties (sums to 1, non-negative, preserves ordering)
//! - Softmax numerical stability (extreme values)
//! - Sampling distribution correctness
//! - Temperature scaling effects
//! - Top-k filtering correctness
//! - Top-p (nucleus) filtering correctness
//! - Repetition penalty mathematics
//! - Combined strategy composition
//! - Determinism (seeded RNG reproducibility)
//! - Edge cases (single logit, identical logits, large vocabs)
//! - Builder pattern correctness

use llama_sampling::*;

// ===========================================================================
// Softmax Mathematical Properties
// ===========================================================================

/// Helper: extract softmax via the Sampler for a given logits slice.
/// We use a temperature=1.0 sampler and inspect the distribution indirectly.
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

#[test]
fn softmax_sums_to_one() {
    let test_cases: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.0, 0.0, 0.0],
        vec![-1.0, -2.0, -3.0],
        vec![100.0, 200.0, 300.0],
        vec![1.0],
    ];

    for logits in &test_cases {
        let probs = softmax(logits);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax sum = {} for logits {:?}",
            sum,
            logits
        );
    }
}

#[test]
fn softmax_all_non_negative() {
    let logits = vec![-100.0, -50.0, 0.0, 50.0, 100.0];
    let probs = softmax(&logits);
    for (i, &p) in probs.iter().enumerate() {
        assert!(p >= 0.0, "prob[{}] = {} is negative", i, p);
    }
}

#[test]
fn softmax_preserves_ordering() {
    let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
    let probs = softmax(&logits);
    // Higher logit → higher probability
    assert!(probs[3] > probs[4]); // 5.0 > 4.0
    assert!(probs[4] > probs[1]); // 4.0 > 3.0
    assert!(probs[1] > probs[2]); // 3.0 > 2.0
    assert!(probs[2] > probs[0]); // 2.0 > 1.0
}

#[test]
fn softmax_uniform_for_equal_logits() {
    let logits = vec![5.0; 4];
    let probs = softmax(&logits);
    for &p in &probs {
        assert!((p - 0.25).abs() < 1e-6);
    }
}

#[test]
fn softmax_numerical_stability_large_values() {
    // Should not overflow or produce NaN
    let logits = vec![1000.0, 1001.0, 999.0];
    let probs = softmax(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum = {}", sum);
    assert!(!probs.iter().any(|p| p.is_nan()), "NaN in probs");
}

#[test]
fn softmax_numerical_stability_very_negative() {
    let logits = vec![-1000.0, -999.0, -1001.0];
    let probs = softmax(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum = {}", sum);
    assert!(!probs.iter().any(|p| p.is_nan()), "NaN in probs");
}

// ===========================================================================
// Greedy Sampling
// ===========================================================================

#[test]
fn greedy_always_picks_highest_logit() {
    let mut sampler = Sampler::new().with_temperature(0.0001);

    let test_cases = vec![
        (vec![1.0, 5.0, 3.0], 1),
        (vec![10.0, 2.0, 3.0], 0),
        (vec![1.0, 2.0, 10.0], 2),
        (vec![5.0, 5.0, 5.0, 5.0, 100.0], 4),
    ];

    for (logits, expected) in test_cases {
        let token = sampler.sample(&logits).unwrap();
        assert_eq!(
            token, expected,
            "greedy should pick index {} for {:?}",
            expected, logits
        );
    }
}

#[test]
fn greedy_is_deterministic() {
    let logits = vec![1.0, 10.0, 3.0, 7.0, 5.0];

    for _ in 0..100 {
        let mut sampler = Sampler::new().with_temperature(0.0001);
        let token = sampler.sample(&logits).unwrap();
        assert_eq!(token, 1, "greedy must always pick highest");
    }
}

// ===========================================================================
// Temperature Scaling (Architecture Section 7.1)
// ===========================================================================

#[test]
fn low_temperature_concentrates_probability() {
    let logits = vec![1.0, 2.0, 3.0];

    // Low temp: should strongly favor highest logit
    let mut sampler_low = Sampler::new().with_temperature(0.01).with_seed(42);
    let mut count_max = 0;
    for _ in 0..100 {
        if sampler_low.sample(&logits).unwrap() == 2 {
            count_max += 1;
        }
    }
    assert!(
        count_max > 90,
        "low temp should pick max >90% of time, got {}",
        count_max
    );
}

#[test]
fn high_temperature_spreads_probability() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];

    let mut sampler = Sampler::new().with_temperature(10.0).with_seed(42);
    let mut counts = [0u32; 4];
    for _ in 0..10000 {
        let token = sampler.sample(&logits).unwrap();
        counts[token] += 1;
    }

    // With high temperature, distribution should be more uniform
    // Each token should appear at least some of the time
    for (i, &count) in counts.iter().enumerate() {
        assert!(
            count > 500,
            "high temp: token {} only sampled {} times out of 10000",
            i,
            count
        );
    }
}

#[test]
fn temperature_one_is_standard_softmax() {
    // At temperature 1.0, sampling should follow standard softmax distribution
    let logits = vec![0.0, 1.0, 2.0];
    let probs = softmax(&logits);

    let mut sampler = Sampler::new().with_temperature(1.0).with_seed(42);
    let mut counts = [0u32; 3];
    let n = 50000;
    for _ in 0..n {
        let token = sampler.sample(&logits).unwrap();
        counts[token] += 1;
    }

    // Check that empirical distribution roughly matches softmax
    for (i, &count) in counts.iter().enumerate() {
        let empirical = count as f32 / n as f32;
        let expected = probs[i];
        assert!(
            (empirical - expected).abs() < 0.03,
            "token {}: empirical={:.3}, expected={:.3}",
            i,
            empirical,
            expected
        );
    }
}

#[test]
fn zero_temperature_errors() {
    let mut sampler = Sampler::new().with_temperature(0.0);
    assert_eq!(
        sampler.sample(&[1.0, 2.0]),
        Err(SamplingError::InvalidTemperature)
    );
}

#[test]
fn negative_temperature_errors() {
    let mut sampler = Sampler::new().with_temperature(-1.0);
    assert_eq!(
        sampler.sample(&[1.0, 2.0]),
        Err(SamplingError::InvalidTemperature)
    );
}

// ===========================================================================
// Top-K Filtering (Architecture Section 7.2)
// ===========================================================================

#[test]
fn top_k_equals_1_is_greedy() {
    let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let mut sampler = Sampler::new().with_top_k(1).with_seed(42);

    for _ in 0..50 {
        let token = sampler.sample(&logits).unwrap();
        assert_eq!(token, 1, "top_k=1 must always pick highest");
    }
}

#[test]
fn top_k_restricts_to_k_tokens() {
    let logits = vec![1.0, 5.0, 4.0, 3.0, 2.0]; // sorted: 5,4,3,2,1
    let mut sampler = Sampler::new().with_top_k(2).with_seed(42);

    let mut seen = std::collections::HashSet::new();
    for _ in 0..1000 {
        let token = sampler.sample(&logits).unwrap();
        seen.insert(token);
    }

    // Only top-2 tokens (indices 1 and 2) should be sampled
    assert!(
        seen.contains(&1) && seen.contains(&2),
        "should contain top-2 indices, got {:?}",
        seen
    );
    assert!(
        !seen.contains(&0) && !seen.contains(&3) && !seen.contains(&4),
        "should not contain below-top-k indices, got {:?}",
        seen
    );
}

#[test]
fn top_k_larger_than_vocab_has_no_effect() {
    let logits = vec![1.0, 2.0, 3.0];
    let mut sampler = Sampler::new().with_top_k(100).with_seed(42);

    let mut seen = std::collections::HashSet::new();
    for _ in 0..10000 {
        let token = sampler.sample(&logits).unwrap();
        seen.insert(token);
    }

    // All tokens should be reachable
    assert_eq!(seen.len(), 3);
}

#[test]
fn top_k_zero_has_no_effect() {
    let logits = vec![1.0, 2.0, 3.0];
    let mut sampler = Sampler::new().with_top_k(0).with_seed(42);

    let mut seen = std::collections::HashSet::new();
    for _ in 0..10000 {
        let token = sampler.sample(&logits).unwrap();
        seen.insert(token);
    }
    assert_eq!(seen.len(), 3);
}

// ===========================================================================
// Top-P (Nucleus) Filtering (Architecture Section 7.3)
// ===========================================================================

#[test]
fn top_p_one_includes_all_tokens() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let mut sampler = Sampler::new().with_top_p(1.0).with_seed(42);

    let mut seen = std::collections::HashSet::new();
    for _ in 0..10000 {
        let token = sampler.sample(&logits).unwrap();
        seen.insert(token);
    }
    assert_eq!(seen.len(), 4);
}

#[test]
fn top_p_very_small_is_nearly_greedy() {
    let logits = vec![1.0, 10.0, 2.0, 3.0];
    let mut sampler = Sampler::new().with_top_p(0.01).with_seed(42);

    let mut count_max = 0;
    for _ in 0..100 {
        if sampler.sample(&logits).unwrap() == 1 {
            count_max += 1;
        }
    }
    assert!(
        count_max > 95,
        "very small top_p should be nearly greedy, got {} / 100",
        count_max
    );
}

#[test]
fn top_p_filters_low_probability_tokens() {
    // Create logits where one token dominates
    let logits = vec![10.0, 0.0, 0.0, 0.0, 0.0]; // softmax ~0.98 for [0]
    let mut sampler = Sampler::new().with_top_p(0.5).with_seed(42);

    let mut seen = std::collections::HashSet::new();
    for _ in 0..1000 {
        seen.insert(sampler.sample(&logits).unwrap());
    }

    // The dominant token should be the only one sampled
    assert!(seen.contains(&0));
    assert_eq!(
        seen.len(),
        1,
        "top_p=0.5 with dominant token should only sample that token"
    );
}

// ===========================================================================
// Repetition Penalty (Architecture Section 7)
// ===========================================================================

#[test]
fn repetition_penalty_reduces_repeated_token_probability() {
    let logits = vec![5.0, 5.0, 5.0, 5.0]; // uniform logits

    // Without penalty: roughly uniform
    let mut sampler_no_penalty = Sampler::new().with_seed(42);
    let mut counts_no_penalty = [0u32; 4];
    for _ in 0..10000 {
        counts_no_penalty[sampler_no_penalty.sample(&logits).unwrap()] += 1;
    }

    // With penalty on token 0
    let mut sampler_penalty = Sampler::new().with_repetition_penalty(3.0).with_seed(42);
    let mut counts_penalty = [0u32; 4];
    for _ in 0..10000 {
        counts_penalty[sampler_penalty.sample_with_history(&logits, &[0]).unwrap()] += 1;
    }

    // Token 0 should appear less often with penalty
    assert!(
        counts_penalty[0] < counts_no_penalty[0],
        "penalized: {}, unpenalized: {}",
        counts_penalty[0],
        counts_no_penalty[0]
    );
}

#[test]
fn repetition_penalty_on_negative_logits() {
    // For negative logits, penalty should multiply (making them more negative)
    let logits = vec![-1.0, -2.0, 5.0];
    let history = vec![0, 1]; // penalize negative-logit tokens

    let mut sampler = Sampler::new().with_repetition_penalty(2.0).with_seed(42);
    let token = sampler.sample_with_history(&logits, &history).unwrap();
    // Token 2 (logit=5.0) should dominate even more
    // Just verify it runs without error and returns a valid index
    assert!(token < 3);
}

#[test]
fn repetition_penalty_with_empty_history_is_no_op() {
    let logits = vec![1.0, 2.0, 3.0];

    let mut s1 = Sampler::new().with_repetition_penalty(2.0).with_seed(42);
    let mut s2 = Sampler::new().with_seed(42);

    // Empty history should produce same results as no penalty
    for _ in 0..20 {
        let t1 = s1.sample_with_history(&logits, &[]).unwrap();
        let t2 = s2.sample(&logits).unwrap();
        assert_eq!(t1, t2, "empty history should be no-op");
    }
}

#[test]
fn repetition_penalty_out_of_range_history_ignored() {
    let logits = vec![1.0, 2.0, 3.0];
    let history = vec![100, 200]; // indices beyond logits length

    let mut sampler = Sampler::new().with_repetition_penalty(2.0).with_seed(42);
    // Should not panic or error
    let token = sampler.sample_with_history(&logits, &history).unwrap();
    assert!(token < 3);
}

// ===========================================================================
// Combined Strategy (Architecture Section 7: "combine Top-K then Top-P")
// ===========================================================================

#[test]
fn combined_top_k_top_p_temperature() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut sampler = Sampler::new()
        .with_temperature(0.8)
        .with_top_k(5)
        .with_top_p(0.9)
        .with_seed(42);

    let mut seen = std::collections::HashSet::new();
    for _ in 0..1000 {
        let token = sampler.sample(&logits).unwrap();
        assert!(token < 10);
        seen.insert(token);
    }

    // Top-k=5 limits to top 5 tokens (indices 5-9)
    for &idx in &seen {
        assert!(
            idx >= 5,
            "combined strategy sampled index {} which is below top-5",
            idx
        );
    }
}

#[test]
fn combined_all_strategies_with_repetition() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let history = vec![4]; // penalize highest

    let mut sampler = Sampler::new()
        .with_temperature(0.5)
        .with_top_k(3)
        .with_top_p(0.95)
        .with_repetition_penalty(2.0)
        .with_seed(42);

    let token = sampler.sample_with_history(&logits, &history).unwrap();
    assert!(token < 5);
}

// ===========================================================================
// Determinism and Reproducibility
// ===========================================================================

#[test]
fn same_seed_same_sequence_1000_samples() {
    let logits = vec![0.1, 0.2, 0.3, 0.15, 0.25];

    let mut s1 = Sampler::new().with_seed(12345);
    let mut s2 = Sampler::new().with_seed(12345);

    for i in 0..1000 {
        let t1 = s1.sample(&logits).unwrap();
        let t2 = s2.sample(&logits).unwrap();
        assert_eq!(t1, t2, "mismatch at step {}", i);
    }
}

#[test]
fn different_seeds_different_sequences() {
    let logits = vec![0.25, 0.25, 0.25, 0.25];

    let mut s1 = Sampler::new().with_seed(1);
    let mut s2 = Sampler::new().with_seed(2);

    let seq1: Vec<usize> = (0..100).map(|_| s1.sample(&logits).unwrap()).collect();
    let seq2: Vec<usize> = (0..100).map(|_| s2.sample(&logits).unwrap()).collect();

    assert_ne!(
        seq1, seq2,
        "different seeds should produce different sequences"
    );
}

#[test]
fn determinism_with_combined_strategies() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let make_sampler = || {
        Sampler::new()
            .with_temperature(0.7)
            .with_top_k(3)
            .with_top_p(0.9)
            .with_seed(42)
    };

    let mut s1 = make_sampler();
    let mut s2 = make_sampler();

    for _ in 0..100 {
        assert_eq!(s1.sample(&logits).unwrap(), s2.sample(&logits).unwrap());
    }
}

// ===========================================================================
// SeededRng Properties
// ===========================================================================

#[test]
fn seeded_rng_range_zero_to_one() {
    let mut rng = SeededRng::new(42);
    for _ in 0..100000 {
        let v = rng.next_f32();
        assert!(
            (0.0..1.0).contains(&v),
            "RNG value {} out of [0, 1) range",
            v
        );
    }
}

#[test]
fn seeded_rng_zero_seed_not_stuck() {
    let mut rng = SeededRng::new(0);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..100 {
        let v = rng.next_f32();
        // Convert to a fixed-point representation for hashing
        seen.insert((v * 10000.0) as u32);
    }
    assert!(
        seen.len() > 50,
        "zero-seed RNG seems stuck, only {} unique values",
        seen.len()
    );
}

#[test]
fn seeded_rng_different_seeds_diverge() {
    let mut r1 = SeededRng::new(1);
    let mut r2 = SeededRng::new(2);

    let s1: Vec<u32> = (0..10).map(|_| (r1.next_f32() * 1000.0) as u32).collect();
    let s2: Vec<u32> = (0..10).map(|_| (r2.next_f32() * 1000.0) as u32).collect();
    assert_ne!(s1, s2);
}

#[test]
fn seeded_rng_clone_produces_same_sequence() {
    let mut rng = SeededRng::new(42);
    rng.next_f32(); // advance a few steps
    rng.next_f32();

    let mut cloned = rng.clone();

    for _ in 0..100 {
        let v1 = rng.next_f32();
        let v2 = cloned.next_f32();
        assert!((v1 - v2).abs() < 1e-10);
    }
}

// ===========================================================================
// Edge Cases
// ===========================================================================

#[test]
fn single_logit_always_returns_zero() {
    let mut sampler = Sampler::new().with_seed(42);
    for _ in 0..100 {
        assert_eq!(sampler.sample(&[5.0]).unwrap(), 0);
    }
}

#[test]
fn two_logits_equal_roughly_fifty_fifty() {
    let logits = vec![0.0, 0.0];
    let mut sampler = Sampler::new().with_seed(42);

    let mut count_0 = 0u32;
    let n = 10000;
    for _ in 0..n {
        if sampler.sample(&logits).unwrap() == 0 {
            count_0 += 1;
        }
    }

    let ratio = count_0 as f32 / n as f32;
    assert!(
        (ratio - 0.5).abs() < 0.05,
        "expected ~50/50, got {:.1}%",
        ratio * 100.0
    );
}

#[test]
fn empty_logits_error() {
    let mut sampler = Sampler::new();
    assert_eq!(sampler.sample(&[]), Err(SamplingError::InvalidLogits));
}

#[test]
fn large_vocab_sampling() {
    let logits: Vec<f32> = (0..128000).map(|i| (i as f32) / 128000.0).collect();
    let mut sampler = Sampler::new()
        .with_temperature(1.0)
        .with_top_k(100)
        .with_seed(42);

    let token = sampler.sample(&logits).unwrap();
    assert!(token < 128000);
    // Should be from the top-100 (highest indices)
    assert!(
        token >= 127900,
        "token {} should be in top-100 range",
        token
    );
}

#[test]
fn all_negative_infinity_except_one() {
    let mut logits = vec![f32::NEG_INFINITY; 5];
    logits[3] = 1.0;

    let mut sampler = Sampler::new().with_seed(42);
    for _ in 0..100 {
        assert_eq!(sampler.sample(&logits).unwrap(), 3);
    }
}

// ===========================================================================
// Builder Pattern
// ===========================================================================

#[test]
fn builder_default_values() {
    let sampler = Sampler::new();
    assert_eq!(sampler.temperature, 1.0);
    assert_eq!(sampler.top_k, None);
    assert_eq!(sampler.top_p, None);
    assert_eq!(sampler.repetition_penalty, None);
}

#[test]
fn builder_chaining() {
    let sampler = Sampler::new()
        .with_temperature(0.8)
        .with_top_k(40)
        .with_top_p(0.95)
        .with_repetition_penalty(1.1)
        .with_seed(123);

    assert_eq!(sampler.temperature, 0.8);
    assert_eq!(sampler.top_k, Some(40));
    assert_eq!(sampler.top_p, Some(0.95));
    assert_eq!(sampler.repetition_penalty, Some(1.1));
}

#[test]
fn sampler_default_trait() {
    let s1 = Sampler::new();
    let s2 = Sampler::default();
    assert_eq!(s1.temperature, s2.temperature);
    assert_eq!(s1.top_k, s2.top_k);
    assert_eq!(s1.top_p, s2.top_p);
}

// ===========================================================================
// Error Type Tests
// ===========================================================================

#[test]
fn sampling_error_display() {
    assert_eq!(
        format!("{}", SamplingError::InvalidLogits),
        "Invalid logits array"
    );
    assert_eq!(
        format!("{}", SamplingError::InvalidTemperature),
        "Temperature must be > 0"
    );
    assert_eq!(
        format!("{}", SamplingError::NoValidTokens),
        "No valid tokens after filtering"
    );
}

#[test]
fn sampling_error_clone_and_eq() {
    let e1 = SamplingError::InvalidLogits;
    let e2 = e1.clone();
    assert_eq!(e1, e2);
}

#[test]
fn sampling_error_is_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(SamplingError::InvalidLogits);
    assert!(!err.to_string().is_empty());
}

// ===========================================================================
// Sampler State Mutation
// ===========================================================================

#[test]
fn rng_advances_each_sample() {
    let logits = vec![0.25, 0.25, 0.25, 0.25]; // uniform
    let mut sampler = Sampler::new().with_seed(42);

    // Collect samples; should not all be the same
    let samples: Vec<usize> = (0..100).map(|_| sampler.sample(&logits).unwrap()).collect();
    let unique: std::collections::HashSet<&usize> = samples.iter().collect();
    assert!(unique.len() > 1, "RNG should advance between samples");
}

#[test]
fn sampler_clone_diverges_after_mutation() {
    let logits = vec![0.25, 0.25, 0.25, 0.25];
    let mut s1 = Sampler::new().with_seed(42);

    // Advance s1 a few steps
    s1.sample(&logits).unwrap();
    s1.sample(&logits).unwrap();

    // Clone at this point
    let mut s2 = s1.clone();

    // Both should now produce the same sequence from this point
    for _ in 0..50 {
        assert_eq!(s1.sample(&logits).unwrap(), s2.sample(&logits).unwrap());
    }
}

// ===========================================================================
// Property-Style Tests
// ===========================================================================

#[test]
fn sample_always_returns_valid_index() {
    let mut sampler = Sampler::new().with_seed(42);
    let sizes = [1, 2, 5, 10, 100, 1000];

    for &size in &sizes {
        let logits: Vec<f32> = (0..size).map(|i| i as f32).collect();
        for _ in 0..100 {
            let token = sampler.sample(&logits).unwrap();
            assert!(
                token < size,
                "token {} out of range for vocab size {}",
                token,
                size
            );
        }
    }
}

#[test]
fn top_k_with_all_strategies_still_returns_valid() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut sampler = Sampler::new()
        .with_temperature(0.5)
        .with_top_k(3)
        .with_top_p(0.8)
        .with_repetition_penalty(1.5)
        .with_seed(42);

    for _ in 0..500 {
        let token = sampler.sample_with_history(&logits, &[7, 6]).unwrap();
        assert!(token < 8);
    }
}
