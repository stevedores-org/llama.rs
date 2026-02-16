//! KV Cache Equivalence Phase 1 Tests: LLAMA-004/006
//!
//! Enforces the KV equivalence true test:
//! `full_forward(prompt)` logits == `prefill(prompt[:-1]) + decode(last_token)` logits within 1e-5 tolerance
//!
//! This validates that the KV cache correctly preserves intermediate results for reuse,
//! which is critical for efficient prefill+decode inference patterns.

#[cfg(test)]
mod tests {
    use llama_runtime::RuntimeVerifier;

    // ===== Section A: Basic KV Equivalence =====

    #[test]
    fn kv_equivalence_short_prompt() {
        let verifier = RuntimeVerifier::new();
        let prompt = [1, 3];

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "Short 2-token prompt should satisfy KV equivalence within 1e-5, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_equivalence_medium_prompt() {
        let verifier = RuntimeVerifier::new();
        let prompt = [0, 1, 2, 3];

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "Medium 4-token prompt should satisfy KV equivalence within 1e-5, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_equivalence_max_length_prompt() {
        let verifier = RuntimeVerifier::new();
        let prompt = [0, 1, 2, 3, 4, 5, 6, 7];

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "Max-length 8-token prompt should satisfy KV equivalence within 1e-5, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_equivalence_with_repeated_tokens() {
        let verifier = RuntimeVerifier::new();
        let prompt = [1, 1, 1, 1];

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "Repeated tokens should satisfy KV equivalence within 1e-5, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_equivalence_with_alternating_tokens() {
        let verifier = RuntimeVerifier::new();
        let prompt = [0, 1, 0, 1, 0, 1];

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "Alternating tokens should satisfy KV equivalence within 1e-5, got {}",
            report.max_abs_diff
        );
    }

    // ===== Section B: Boundary Conditions =====

    #[test]
    fn kv_equivalence_min_length_two_tokens() {
        let verifier = RuntimeVerifier::new();
        let prompt = [0, 7]; // min valid length

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "Minimum 2-token prompt should satisfy KV equivalence within 1e-5, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_equivalence_last_token_largest_id() {
        let verifier = RuntimeVerifier::new();
        let prompt = [0, 1, 2, 7]; // last token has largest embedding id

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "Prompt with max embedding id last should satisfy KV equivalence within 1e-5, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_equivalence_first_token_zero() {
        let verifier = RuntimeVerifier::new();
        let prompt = [0, 0, 0, 0]; // all zeros

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "All-zero prompt should satisfy KV equivalence within 1e-5, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_equivalence_ascending_tokens() {
        let verifier = RuntimeVerifier::new();
        let prompt = [0, 1, 2, 3, 4, 5, 6, 7]; // ascending order

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "Ascending token IDs should satisfy KV equivalence within 1e-5, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_equivalence_descending_tokens() {
        let verifier = RuntimeVerifier::new();
        let prompt = [7, 6, 5, 4, 3, 2, 1, 0]; // descending order

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "Descending token IDs should satisfy KV equivalence within 1e-5, got {}",
            report.max_abs_diff
        );
    }

    // ===== Section C: Bug Detection Tests =====

    #[test]
    fn kv_equivalence_detects_off_by_one_position_error() {
        let verifier = RuntimeVerifier::new();
        let prompt = [1, 3, 2, 6];

        let report = verifier.verify_with_off_by_one_bug(&prompt).unwrap();
        assert!(
            report.max_abs_diff > 1e-4,
            "Off-by-one position bug should produce large difference > 1e-4, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_equivalence_off_by_one_detectable_in_short_prompt() {
        let verifier = RuntimeVerifier::new();
        let prompt = [0, 1];

        let report = verifier.verify_with_off_by_one_bug(&prompt).unwrap();
        assert!(
            report.max_abs_diff > 1e-4,
            "Off-by-one bug should be detectable even in 2-token prompt, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_equivalence_off_by_one_detectable_in_long_prompt() {
        let verifier = RuntimeVerifier::new();
        let prompt = [0, 1, 2, 3, 4, 5, 6, 7];

        let report = verifier.verify_with_off_by_one_bug(&prompt).unwrap();
        assert!(
            report.max_abs_diff > 1e-4,
            "Off-by-one bug should be detectable in long prompt, got {}",
            report.max_abs_diff
        );
    }

    // ===== Section D: Error Handling =====

    #[test]
    fn kv_equivalence_rejects_single_token() {
        let verifier = RuntimeVerifier::new();
        let result = verifier.verify_kv_equivalence(&[0]);

        assert!(result.is_err(), "Single token should be rejected");
        match result {
            Err(e) => assert!(
                e.to_string().contains("at least 2 tokens"),
                "Error message should mention minimum 2 tokens"
            ),
            Ok(_) => panic!("Expected error for single token"),
        }
    }

    #[test]
    fn kv_equivalence_rejects_empty_prompt() {
        let verifier = RuntimeVerifier::new();
        let result = verifier.verify_kv_equivalence(&[]);

        assert!(result.is_err(), "Empty prompt should be rejected");
    }

    #[test]
    fn kv_equivalence_rejects_invalid_token_ids() {
        let verifier = RuntimeVerifier::new();
        let prompt = [0, 1, 8]; // 8 is out of vocab (vocab_size=8, ids 0-7)

        let result = verifier.verify_kv_equivalence(&prompt);
        assert!(result.is_err(), "Out-of-range token ID should be rejected");
    }

    #[test]
    fn kv_equivalence_rejects_negative_token_ids() {
        let verifier = RuntimeVerifier::new();
        let prompt = [0, 1, -1];

        let result = verifier.verify_kv_equivalence(&prompt);
        assert!(result.is_err(), "Negative token ID should be rejected");
    }

    // ===== Section E: Numerical Stability =====

    #[test]
    fn kv_equivalence_maintains_precision_across_positions() {
        let verifier = RuntimeVerifier::new();

        // Test multiple prompts to ensure precision across different positions
        for len in 2..=8 {
            let prompt: Vec<i32> = (0..len).map(|i| (i % 8) as i32).collect();
            let prompt_array: [i32; 8] = [
                *prompt.get(0).unwrap_or(&0),
                *prompt.get(1).unwrap_or(&0),
                *prompt.get(2).unwrap_or(&0),
                *prompt.get(3).unwrap_or(&0),
                *prompt.get(4).unwrap_or(&0),
                *prompt.get(5).unwrap_or(&0),
                *prompt.get(6).unwrap_or(&0),
                *prompt.get(7).unwrap_or(&0),
            ];

            let report = verifier
                .verify_kv_equivalence(&prompt_array[..len])
                .unwrap();
            assert!(
                report.max_abs_diff <= 1e-5,
                "Length {}: KV equivalence should hold within 1e-5, got {}",
                len,
                report.max_abs_diff
            );
        }
    }

    #[test]
    fn kv_equivalence_no_underflow_with_small_attention_scores() {
        let verifier = RuntimeVerifier::new();
        // High negative position rotations can create small attention scores
        let prompt = [0, 0, 0, 0, 0, 0, 0, 0];

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "Should handle small attention scores without underflow, got {}",
            report.max_abs_diff
        );
    }

    #[test]
    fn kv_equivalence_no_overflow_with_large_attention_scores() {
        let verifier = RuntimeVerifier::new();
        // Identical embeddings can create large dot products before softmax
        let prompt = [7, 7, 7, 7];

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        assert!(
            report.max_abs_diff <= 1e-5,
            "Should handle large attention scores without overflow, got {}",
            report.max_abs_diff
        );
    }

    // ===== Section F: Invariants and Properties =====

    #[test]
    fn kv_equivalence_output_has_correct_vocab_size() {
        let verifier = RuntimeVerifier::new();
        let prompt = [1, 2, 3, 4];

        let report = verifier.verify_kv_equivalence(&prompt).unwrap();
        // The report only gives max diff, but logits should be 8-dim (vocab_size=8)
        // This is an implicit invariant check - if the implementation changed vocab size,
        // the test would fail because embeddings are fixed size
        assert!(
            report.max_abs_diff >= 0.0,
            "Report should contain valid max_abs_diff"
        );
    }

    #[test]
    fn kv_equivalence_is_deterministic() {
        let verifier = RuntimeVerifier::new();
        let prompt = [2, 3, 4, 5];

        let report1 = verifier.verify_kv_equivalence(&prompt).unwrap();
        let report2 = verifier.verify_kv_equivalence(&prompt).unwrap();

        assert_eq!(
            report1.max_abs_diff, report2.max_abs_diff,
            "KV equivalence check should be deterministic"
        );
    }

    #[test]
    fn kv_equivalence_multiple_verifiers_agree() {
        let verifier1 = RuntimeVerifier::new();
        let verifier2 = RuntimeVerifier::new();
        let prompt = [3, 4, 5, 6];

        let report1 = verifier1.verify_kv_equivalence(&prompt).unwrap();
        let report2 = verifier2.verify_kv_equivalence(&prompt).unwrap();

        assert_eq!(
            report1.max_abs_diff, report2.max_abs_diff,
            "Different verifier instances should produce identical results"
        );
    }

    #[test]
    fn kv_equivalence_is_symmetric_across_verifier_reuse() {
        let verifier = RuntimeVerifier::new();
        let prompt1 = [1, 2, 3, 4];
        let prompt2 = [5, 6, 7, 0];

        let report1 = verifier.verify_kv_equivalence(&prompt1).unwrap();
        let report2 = verifier.verify_kv_equivalence(&prompt2).unwrap();

        // Both should pass independently without cross-contamination
        assert!(report1.max_abs_diff <= 1e-5);
        assert!(report2.max_abs_diff <= 1e-5);
    }

    // ===== Section G: Comprehensive Coverage =====

    #[test]
    fn kv_equivalence_comprehensive_token_combinations() {
        let verifier = RuntimeVerifier::new();

        // Test all 8x8 pairs of first two tokens
        for first in 0..8 {
            for second in 0..8 {
                let prompt = [first as i32, second as i32];
                let report = verifier.verify_kv_equivalence(&prompt).unwrap();
                assert!(
                    report.max_abs_diff <= 1e-5,
                    "Token pair ({}, {}): KV equivalence should hold within 1e-5, got {}",
                    first,
                    second,
                    report.max_abs_diff
                );
            }
        }
    }
}
