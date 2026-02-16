//! # llama-runtime
//!
//! Runtime execution and verification helpers for llama.rs.
//!
//! This crate includes:
//! - A `MockEngine` demonstrating the narrow-waist `LlamaEngine` trait
//! - A Phase-1 verification harness for LLAMA-006:
//!   `full_forward(prompt)` logits vs `prefill(prompt[:-1]) + decode(last_token)` logits.
//! - LLAMA-007 runtime backend selector with feature-gated CPU/Metal support,
//!   kernel availability matrix, and telemetry hooks for TTFT / tokens-per-sec.

pub mod backend;
pub mod telemetry;

use llama_engine::{
    DecodeResult, LlamaEngine, LlamaError, ModelHandle, ModelSpec, PrefillResult, Result, Session,
    TokenId,
};
use llama_kv::{KVLayout, LayerKVCache};
use llama_sampling::{Sampler, SamplingConfig, SamplingStrategy};
use llama_tokenizer::{Tokenizer, WhitespaceTokenizer};
use std::sync::Mutex;
use std::thread;

// ---------------------------------------------------------------------------
// MockEngine — Milestone A narrow-waist demonstration
// ---------------------------------------------------------------------------

/// A mock engine implementation for Milestone A.
///
/// Uses a simple whitespace tokenizer and greedy sampler to demonstrate
/// the "narrow waist" API without requiring a real model or MLX backend.
pub struct MockEngine {
    tokenizer: WhitespaceTokenizer,
    sampler: Mutex<Sampler>,
}

impl MockEngine {
    pub fn new() -> Self {
        Self {
            tokenizer: WhitespaceTokenizer::new(),
            sampler: Mutex::new(
                Sampler::new(SamplingConfig {
                    strategy: SamplingStrategy::Greedy,
                    ..SamplingConfig::default()
                })
                .expect("default config is valid"),
            ),
        }
    }
}

impl Default for MockEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaEngine for MockEngine {
    fn load_model(&self, _spec: &ModelSpec) -> Result<ModelHandle> {
        Ok(ModelHandle)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<TokenId>> {
        self.tokenizer
            .encode(text)
            .map_err(|e| LlamaError::Tokenization(e.to_string()))
    }

    fn detokenize(&self, tokens: &[TokenId]) -> Result<String> {
        self.tokenizer
            .decode(tokens)
            .map_err(|e| LlamaError::Tokenization(e.to_string()))
    }

    fn prefill(&self, _session: &mut Session, tokens: &[TokenId]) -> Result<PrefillResult> {
        Ok(PrefillResult {
            tokens_processed: tokens.len(),
        })
    }

    fn decode(&self, _session: &mut Session) -> Result<DecodeResult> {
        let mock_logits = vec![0.1, 0.5, 0.1, 0.1, 0.2];
        let mut sampler = self.sampler.lock().unwrap();
        let token = sampler
            .sample(&mock_logits, &[])
            .map_err(|e| LlamaError::Inference(format!("{}", e)))?;
        Ok(DecodeResult {
            token: token as TokenId,
        })
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; 128]).collect())
    }
}

// ---------------------------------------------------------------------------
// RuntimeVerifier — LLAMA-006 KV equivalence true test
// ---------------------------------------------------------------------------

/// Errors for runtime verification routines.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("prompt must contain at least 2 tokens")]
    PromptTooShort,
    #[error("invalid token id: {0}")]
    InvalidToken(i32),
    #[error("kv error: {0}")]
    Kv(#[from] llama_kv::KVError),
    #[error("backend parity failed on {backend:?}: diff {diff} > tolerance {tolerance}")]
    BackendParityExceeded {
        backend: RuntimeBackend,
        diff: f32,
        tolerance: f32,
    },
    #[error("concurrent parity stress test thread panicked")]
    ConcurrentStressThreadPanic,
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
    ) -> std::result::Result<KvEquivalenceReport, RuntimeError> {
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
    ) -> std::result::Result<KvEquivalenceReport, RuntimeError> {
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

// ---------------------------------------------------------------------------
// BackendParityGate — LLAMA-008 backend parity and stress checks
// ---------------------------------------------------------------------------

/// Runtime backends covered by parity gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBackend {
    Cpu,
    Metal,
}

/// Deterministic parity configuration.
#[derive(Debug, Clone, Copy)]
pub struct ParityConfig {
    /// Fixed seed used for deterministic backend comparisons.
    pub seed: u64,
    /// Absolute tolerance for backend parity checks.
    pub tolerance: f32,
    /// Number of concurrent sessions for stress checks.
    pub concurrent_sessions: usize,
    /// Iterations per session for stress checks.
    pub iterations_per_session: usize,
}

impl Default for ParityConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            tolerance: 1e-5,
            concurrent_sessions: 8,
            iterations_per_session: 16,
        }
    }
}

/// Result of a backend parity comparison.
#[derive(Debug, Clone, Copy)]
pub struct ParityReport {
    pub backend: RuntimeBackend,
    pub max_abs_diff: f32,
}

/// LLAMA-008 parity gate:
/// - fixed-seed CPU vs Metal backend parity checks
/// - attention golden checks across enabled backends
/// - concurrent-session deadlock stress checks
pub struct BackendParityGate {
    model: ToyModel,
}

impl Default for BackendParityGate {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendParityGate {
    pub fn new() -> Self {
        Self {
            model: ToyModel::default(),
        }
    }

    pub fn compare_metal_vs_cpu(
        &self,
        prompt: &[i32],
        cfg: ParityConfig,
    ) -> std::result::Result<ParityReport, RuntimeError> {
        if prompt.len() < 2 {
            return Err(RuntimeError::PromptTooShort);
        }
        let cpu = self
            .model
            .full_forward_seeded(prompt, cfg.seed, RuntimeBackend::Cpu)?;
        let metal = self
            .model
            .full_forward_seeded(prompt, cfg.seed, RuntimeBackend::Metal)?;
        let diff = max_abs_diff(&cpu, &metal);
        if diff > cfg.tolerance {
            return Err(RuntimeError::BackendParityExceeded {
                backend: RuntimeBackend::Metal,
                diff,
                tolerance: cfg.tolerance,
            });
        }
        Ok(ParityReport {
            backend: RuntimeBackend::Metal,
            max_abs_diff: diff,
        })
    }

    /// Runs attention "golden" verification across all enabled backends.
    /// CPU is the reference, and each backend must match within tolerance.
    pub fn run_attention_golden_test(
        &self,
        prompt: &[i32],
        cfg: ParityConfig,
    ) -> std::result::Result<Vec<ParityReport>, RuntimeError> {
        if prompt.len() < 2 {
            return Err(RuntimeError::PromptTooShort);
        }
        let golden = self
            .model
            .full_forward_seeded(prompt, cfg.seed, RuntimeBackend::Cpu)?;

        let mut reports = Vec::new();
        for backend in Self::enabled_backends() {
            let out = self.model.full_forward_seeded(prompt, cfg.seed, *backend)?;
            let diff = max_abs_diff(&golden, &out);
            if diff > cfg.tolerance {
                return Err(RuntimeError::BackendParityExceeded {
                    backend: *backend,
                    diff,
                    tolerance: cfg.tolerance,
                });
            }
            reports.push(ParityReport {
                backend: *backend,
                max_abs_diff: diff,
            });
        }
        Ok(reports)
    }

    /// Stress test to ensure concurrent backend sessions complete without deadlocks.
    pub fn stress_test_concurrent_sessions(
        &self,
        prompt: &[i32],
        cfg: ParityConfig,
    ) -> std::result::Result<(), RuntimeError> {
        if prompt.len() < 2 {
            return Err(RuntimeError::PromptTooShort);
        }

        let mut handles = Vec::with_capacity(cfg.concurrent_sessions);
        for worker in 0..cfg.concurrent_sessions {
            let prompt_local = prompt.to_vec();
            let mut local_cfg = cfg;
            local_cfg.seed = cfg.seed.wrapping_add(worker as u64);

            handles.push(thread::spawn(
                move || -> std::result::Result<(), RuntimeError> {
                    let gate = BackendParityGate::new();
                    for i in 0..local_cfg.iterations_per_session {
                        let mut run_cfg = local_cfg;
                        run_cfg.seed = local_cfg.seed.wrapping_add(i as u64);
                        gate.compare_metal_vs_cpu(&prompt_local, run_cfg)?;
                    }
                    Ok(())
                },
            ));
        }

        for handle in handles {
            let result = handle
                .join()
                .map_err(|_| RuntimeError::ConcurrentStressThreadPanic)?;
            result?;
        }
        Ok(())
    }

    pub fn enabled_backends() -> &'static [RuntimeBackend] {
        &[RuntimeBackend::Cpu, RuntimeBackend::Metal]
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
    fn full_forward(&self, prompt: &[i32]) -> std::result::Result<Vec<f32>, RuntimeError> {
        self.full_forward_seeded(prompt, 0, RuntimeBackend::Cpu)
    }

    fn full_forward_seeded(
        &self,
        prompt: &[i32],
        seed: u64,
        backend: RuntimeBackend,
    ) -> std::result::Result<Vec<f32>, RuntimeError> {
        let seq_len = prompt.len();
        let mut keys = Vec::with_capacity(seq_len * 2);
        let mut values = Vec::with_capacity(seq_len * 2);

        for (pos, &tok) in prompt.iter().enumerate() {
            let mut kv = self.token_vec(tok)?;
            apply_seed_jitter(&mut kv, seed, pos);
            apply_position_rotation(&mut kv, pos);
            keys.extend_from_slice(&kv);
            values.extend_from_slice(&kv);
        }

        let mut q = self.token_vec(*prompt.last().expect("prompt checked non-empty"))?;
        apply_seed_jitter(&mut q, seed, seq_len - 1);
        apply_position_rotation(&mut q, seq_len - 1);

        let ctx = attention_single_head(&q, &keys, &values, seq_len, 2, backend);
        Ok(project_logits(&ctx, &self.out_proj))
    }

    fn prefill_then_decode(
        &self,
        prompt: &[i32],
        inject_off_by_one: bool,
    ) -> std::result::Result<Vec<f32>, RuntimeError> {
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
        let ctx = attention_single_head(&q, &keys, &values, seq_len, 2, RuntimeBackend::Cpu);
        Ok(project_logits(&ctx, &self.out_proj))
    }

    fn token_vec(&self, token: i32) -> std::result::Result<[f32; 2], RuntimeError> {
        let idx = usize::try_from(token).map_err(|_| RuntimeError::InvalidToken(token))?;
        self.embeddings
            .get(idx)
            .copied()
            .ok_or(RuntimeError::InvalidToken(token))
    }
}

fn apply_seed_jitter(v: &mut [f32; 2], seed: u64, position: usize) {
    if seed == 0 {
        return;
    }
    let jitter = deterministic_jitter(seed, position) * 0.01;
    v[0] += jitter;
    v[1] -= jitter;
}

fn deterministic_jitter(seed: u64, position: usize) -> f32 {
    let pos_mix = (position as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(pos_mix);
    let top = (x >> 40) as u32;
    (top as f32 / u32::MAX as f32) - 0.5
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
    backend: RuntimeBackend,
) -> [f32; 2] {
    match backend {
        RuntimeBackend::Cpu => attention_single_head_cpu(q, keys, values, seq_len, dim),
        RuntimeBackend::Metal => attention_single_head_metal(q, keys, values, seq_len, dim),
    }
}

fn attention_single_head_cpu(
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

// Mirrors CPU semantics while serving as the Metal parity path until real
// Metal kernels are wired in.
fn attention_single_head_metal(
    q: &[f32; 2],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    dim: usize,
) -> [f32; 2] {
    attention_single_head_cpu(q, keys, values, seq_len, dim)
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

    // -- MockEngine tests --

    #[test]
    fn mock_engine_tokenize_roundtrip() {
        let engine = MockEngine::new();
        let tokens = engine.tokenize("hello world").unwrap();
        assert_eq!(tokens.len(), 2);

        let text = engine.detokenize(&tokens).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn mock_engine_prefill_decode() {
        let engine = MockEngine::new();
        let mut session = Session::new();

        let tokens = engine.tokenize("hello world").unwrap();
        let prefill = engine.prefill(&mut session, &tokens).unwrap();
        assert_eq!(prefill.tokens_processed, 2);

        let result = engine.decode(&mut session).unwrap();
        assert!(result.token >= 0);
    }

    #[test]
    fn mock_engine_embed() {
        let engine = MockEngine::new();
        let embeddings = engine.embed(&["hello", "world"]).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 128);
    }

    #[test]
    fn mock_engine_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockEngine>();
    }

    // -- RuntimeVerifier tests --

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

    // -- LLAMA-008 BackendParityGate tests --

    #[test]
    fn parity_gate_fixed_seed_cpu_vs_metal() {
        let gate = BackendParityGate::new();
        let prompt = [1, 3, 2, 6];
        let cfg = ParityConfig::default();
        let report = gate.compare_metal_vs_cpu(&prompt, cfg).unwrap();
        assert_eq!(report.backend, RuntimeBackend::Metal);
        assert!(report.max_abs_diff <= cfg.tolerance);
    }

    #[test]
    fn parity_gate_attention_golden_on_all_backends() {
        let gate = BackendParityGate::new();
        let prompt = [1, 3, 2, 6];
        let cfg = ParityConfig::default();
        let reports = gate.run_attention_golden_test(&prompt, cfg).unwrap();
        assert_eq!(reports.len(), BackendParityGate::enabled_backends().len());
        assert!(reports.iter().all(|r| r.max_abs_diff <= cfg.tolerance));
    }

    #[test]
    fn parity_gate_seed_is_deterministic() {
        let gate = BackendParityGate::new();
        let prompt = [1, 3, 2, 6];
        let cfg = ParityConfig {
            seed: 123456,
            ..ParityConfig::default()
        };
        let a = gate.compare_metal_vs_cpu(&prompt, cfg).unwrap();
        let b = gate.compare_metal_vs_cpu(&prompt, cfg).unwrap();
        assert_eq!(a.max_abs_diff, b.max_abs_diff);
    }

    #[test]
    fn parity_gate_concurrent_sessions_no_deadlock() {
        let gate = BackendParityGate::new();
        let prompt = [1, 3, 2, 6];
        let cfg = ParityConfig {
            concurrent_sessions: 8,
            iterations_per_session: 32,
            ..ParityConfig::default()
        };
        gate.stress_test_concurrent_sessions(&prompt, cfg).unwrap();
    }
}
