//! Telemetry hooks for inference performance measurement.
//!
//! Provides:
//! - [`InferenceMetrics`] — TTFT, tokens/sec, and generation summary
//! - [`TelemetryHook`] trait — callback interface for real-time metric reporting
//! - [`InferenceTimer`] — records timestamps and computes metrics
//! - [`NoopTelemetry`] / [`LogTelemetry`] — built-in hook implementations

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::backend::Backend;

/// Aggregate metrics from a generation run.
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Backend used for this run.
    pub backend: Backend,
    /// Time to first token in milliseconds (prefill latency).
    pub ttft_ms: f64,
    /// Tokens generated per second (decode throughput, excludes prefill).
    pub tokens_per_sec: f64,
    /// Number of prompt tokens processed during prefill.
    pub prompt_tokens: usize,
    /// Number of tokens generated during decode.
    pub generated_tokens: usize,
    /// Total wall-clock time in milliseconds (prefill + decode).
    pub total_time_ms: f64,
}

/// Callback trait for real-time inference telemetry.
///
/// Implementations receive events at key points during generation.
/// All methods have default no-op implementations so hooks can be selective.
pub trait TelemetryHook: Send + Sync {
    /// Called after prefill completes. `ttft_ms` is time from start to first token ready.
    fn on_prefill_complete(&self, _ttft_ms: f64) {}

    /// Called after each decode step produces a token.
    fn on_token_generated(&self, _token_idx: usize, _elapsed_ms: f64) {}

    /// Called when generation finishes with the full metrics summary.
    fn on_generation_complete(&self, _metrics: &InferenceMetrics) {}
}

/// No-op telemetry hook — zero overhead when metrics aren't needed.
#[derive(Debug, Clone, Copy)]
pub struct NoopTelemetry;

impl TelemetryHook for NoopTelemetry {}

/// Logging telemetry hook — collects metrics into a retrievable report.
#[derive(Debug, Clone)]
pub struct LogTelemetry {
    last_report: Arc<Mutex<Option<InferenceMetrics>>>,
}

impl Default for LogTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl LogTelemetry {
    pub fn new() -> Self {
        Self {
            last_report: Arc::new(Mutex::new(None)),
        }
    }

    /// Retrieve the last completed generation's metrics.
    pub fn last_metrics(&self) -> Option<InferenceMetrics> {
        self.last_report.lock().unwrap().clone()
    }
}

impl TelemetryHook for LogTelemetry {
    fn on_generation_complete(&self, metrics: &InferenceMetrics) {
        *self.last_report.lock().unwrap() = Some(metrics.clone());
    }
}

/// Records timestamps during inference to compute [`InferenceMetrics`].
///
/// Usage:
/// 1. Call [`InferenceTimer::new`] at generation start
/// 2. Call [`mark_prefill_complete`] after prefill
/// 3. Call [`mark_token`] after each decode step
/// 4. Call [`finish`] to compute final metrics
pub struct InferenceTimer {
    backend: Backend,
    prompt_tokens: usize,
    start: Instant,
    prefill_end: Option<Instant>,
    token_count: usize,
    hook: Box<dyn TelemetryHook>,
}

impl InferenceTimer {
    /// Start a new timer for a generation run.
    pub fn new(backend: Backend, prompt_tokens: usize, hook: Box<dyn TelemetryHook>) -> Self {
        Self {
            backend,
            prompt_tokens,
            start: Instant::now(),
            prefill_end: None,
            token_count: 0,
            hook,
        }
    }

    /// Mark prefill phase complete. Fires `on_prefill_complete` hook.
    pub fn mark_prefill_complete(&mut self) {
        let now = Instant::now();
        self.prefill_end = Some(now);
        let ttft_ms = now.duration_since(self.start).as_secs_f64() * 1000.0;
        self.hook.on_prefill_complete(ttft_ms);
    }

    /// Mark a decode token generated. Fires `on_token_generated` hook.
    pub fn mark_token(&mut self) {
        self.token_count += 1;
        let elapsed_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        self.hook.on_token_generated(self.token_count, elapsed_ms);
    }

    /// Finalize and return metrics. Fires `on_generation_complete` hook.
    pub fn finish(self) -> InferenceMetrics {
        let total_time_ms = self.start.elapsed().as_secs_f64() * 1000.0;

        let ttft_ms = self
            .prefill_end
            .map(|t| t.duration_since(self.start).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let decode_time_ms = total_time_ms - ttft_ms;
        let tokens_per_sec = if decode_time_ms > 0.0 && self.token_count > 0 {
            self.token_count as f64 / (decode_time_ms / 1000.0)
        } else {
            0.0
        };

        let metrics = InferenceMetrics {
            backend: self.backend,
            ttft_ms,
            tokens_per_sec,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.token_count,
            total_time_ms,
        };

        self.hook.on_generation_complete(&metrics);
        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_telemetry_compiles_and_runs() {
        let hook = NoopTelemetry;
        hook.on_prefill_complete(10.0);
        hook.on_token_generated(1, 15.0);
        hook.on_generation_complete(&InferenceMetrics {
            backend: Backend::Cpu,
            ttft_ms: 10.0,
            tokens_per_sec: 100.0,
            prompt_tokens: 5,
            generated_tokens: 10,
            total_time_ms: 110.0,
        });
    }

    #[test]
    fn log_telemetry_captures_metrics() {
        let hook = LogTelemetry::new();
        assert!(hook.last_metrics().is_none());

        let metrics = InferenceMetrics {
            backend: Backend::Cpu,
            ttft_ms: 12.5,
            tokens_per_sec: 80.0,
            prompt_tokens: 4,
            generated_tokens: 8,
            total_time_ms: 112.5,
        };
        hook.on_generation_complete(&metrics);

        let captured = hook.last_metrics().unwrap();
        assert_eq!(captured.ttft_ms, 12.5);
        assert_eq!(captured.generated_tokens, 8);
    }

    #[test]
    fn inference_timer_basic_flow() {
        let mut timer = InferenceTimer::new(Backend::Cpu, 3, Box::new(NoopTelemetry));

        timer.mark_prefill_complete();
        timer.mark_token();
        timer.mark_token();
        timer.mark_token();

        let metrics = timer.finish();
        assert_eq!(metrics.backend, Backend::Cpu);
        assert_eq!(metrics.prompt_tokens, 3);
        assert_eq!(metrics.generated_tokens, 3);
        assert!(metrics.ttft_ms >= 0.0);
        assert!(metrics.total_time_ms >= metrics.ttft_ms);
    }

    #[test]
    fn inference_timer_fires_hooks() {
        let log = LogTelemetry::new();

        let mut timer = InferenceTimer::new(Backend::Cpu, 2, Box::new(log.clone()));
        timer.mark_prefill_complete();
        timer.mark_token();
        timer.mark_token();
        let metrics = timer.finish();

        assert_eq!(metrics.generated_tokens, 2);
        assert_eq!(metrics.prompt_tokens, 2);
        assert!(metrics.ttft_ms >= 0.0);

        // LogTelemetry should have captured the report via on_generation_complete.
        let captured = log.last_metrics().unwrap();
        assert_eq!(captured.generated_tokens, 2);
        assert_eq!(captured.prompt_tokens, 2);
    }

    #[test]
    fn inference_timer_no_prefill_mark() {
        let timer = InferenceTimer::new(Backend::Cpu, 1, Box::new(NoopTelemetry));
        let metrics = timer.finish();
        assert_eq!(metrics.ttft_ms, 0.0);
        assert_eq!(metrics.generated_tokens, 0);
    }

    #[test]
    fn inference_metrics_debug_display() {
        let metrics = InferenceMetrics {
            backend: Backend::Cpu,
            ttft_ms: 5.0,
            tokens_per_sec: 150.0,
            prompt_tokens: 10,
            generated_tokens: 20,
            total_time_ms: 138.3,
        };
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("ttft_ms"));
        assert!(debug.contains("tokens_per_sec"));
    }
}
