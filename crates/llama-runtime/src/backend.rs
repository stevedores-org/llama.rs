//! Backend selection and kernel availability for llama.rs.
//!
//! Provides:
//! - [`Backend`] enum gated by cargo features (`cpu`, `metal`)
//! - [`KernelOp`] — operations a backend can accelerate
//! - [`KernelMatrix`] — probes which ops each backend supports
//! - [`BackendSelector`] — picks the best available backend at startup

use std::collections::HashMap;
use std::fmt;

/// Compute backend for inference.
///
/// Variants are compile-time gated by the `cpu` and `metal` cargo features.
/// At least one feature must be enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    #[cfg(feature = "cpu")]
    Cpu,
    #[cfg(feature = "metal")]
    Metal,
}

impl Backend {
    /// All backends enabled at compile time.
    pub fn compiled() -> &'static [Backend] {
        &[
            #[cfg(feature = "cpu")]
            Backend::Cpu,
            #[cfg(feature = "metal")]
            Backend::Metal,
        ]
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "cpu")]
            Backend::Cpu => write!(f, "cpu"),
            #[cfg(feature = "metal")]
            Backend::Metal => write!(f, "metal"),
        }
    }
}

/// Operations that a backend can accelerate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelOp {
    /// RMS normalization.
    RmsNorm,
    /// Rotary positional embeddings.
    Rope,
    /// Scaled dot-product attention (decode step).
    Attention,
    /// SwiGLU MLP.
    MlpSwiGlu,
    /// Token embedding lookup.
    Embedding,
    /// Output projection to vocabulary logits.
    Projection,
}

impl KernelOp {
    /// All defined kernel operations.
    pub fn all() -> &'static [KernelOp] {
        &[
            KernelOp::RmsNorm,
            KernelOp::Rope,
            KernelOp::Attention,
            KernelOp::MlpSwiGlu,
            KernelOp::Embedding,
            KernelOp::Projection,
        ]
    }
}

impl fmt::Display for KernelOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelOp::RmsNorm => write!(f, "rms_norm"),
            KernelOp::Rope => write!(f, "rope"),
            KernelOp::Attention => write!(f, "attention"),
            KernelOp::MlpSwiGlu => write!(f, "mlp_swiglu"),
            KernelOp::Embedding => write!(f, "embedding"),
            KernelOp::Projection => write!(f, "projection"),
        }
    }
}

/// Errors from backend selection and validation.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("no backends compiled (enable `cpu` or `metal` feature)")]
    NoBackendsCompiled,
    #[error("backend {backend} missing support for: {}", missing.iter().map(|o| o.to_string()).collect::<Vec<_>>().join(", "))]
    UnsupportedOps {
        backend: Backend,
        missing: Vec<KernelOp>,
    },
    #[error("backend {0} not compiled (missing cargo feature)")]
    NotCompiled(String),
}

/// Kernel availability matrix: maps (backend, op) → supported.
///
/// Call [`KernelMatrix::probe`] at startup to discover which ops
/// each compiled backend can execute.
#[derive(Debug, Clone)]
pub struct KernelMatrix {
    support: HashMap<(Backend, KernelOp), bool>,
}

impl KernelMatrix {
    /// Probe all compiled backends for operation support.
    ///
    /// CPU supports all ops. Metal currently supports all ops on macOS
    /// (placeholder until real Metal kernel probing is wired in).
    pub fn probe() -> Self {
        let mut support = HashMap::new();

        for &backend in Backend::compiled() {
            for &op in KernelOp::all() {
                let supported = match backend {
                    #[cfg(feature = "cpu")]
                    Backend::Cpu => true,
                    #[cfg(feature = "metal")]
                    Backend::Metal => Self::probe_metal_op(op),
                };
                support.insert((backend, op), supported);
            }
        }

        Self { support }
    }

    /// Check if a specific (backend, op) pair is supported.
    pub fn supports(&self, backend: Backend, op: KernelOp) -> bool {
        self.support.get(&(backend, op)).copied().unwrap_or(false)
    }

    /// Validate that a backend supports all required ops.
    /// Returns `Ok(())` or an error listing unsupported ops.
    pub fn validate(&self, backend: Backend) -> Result<(), BackendError> {
        let missing: Vec<KernelOp> = KernelOp::all()
            .iter()
            .filter(|&&op| !self.supports(backend, op))
            .copied()
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(BackendError::UnsupportedOps { backend, missing })
        }
    }

    /// Return all ops supported by a given backend.
    pub fn supported_ops(&self, backend: Backend) -> Vec<KernelOp> {
        KernelOp::all()
            .iter()
            .filter(|&&op| self.supports(backend, op))
            .copied()
            .collect()
    }

    /// Metal op probing placeholder.
    ///
    /// On macOS, Metal is assumed available for all ops (until real
    /// kernel discovery via `MTLDevice` is integrated). On non-macOS,
    /// Metal ops are unsupported.
    #[cfg(feature = "metal")]
    fn probe_metal_op(_op: KernelOp) -> bool {
        cfg!(target_os = "macos")
    }
}

/// Selects and validates the active inference backend.
///
/// Use [`BackendSelector::auto`] for automatic best-backend detection,
/// or [`BackendSelector::with_backend`] to force a specific backend.
#[derive(Debug, Clone)]
pub struct BackendSelector {
    active: Backend,
    matrix: KernelMatrix,
}

impl BackendSelector {
    /// Automatically select the best available backend.
    ///
    /// Preference order: Metal (if on macOS + compiled) > CPU.
    /// Validates that the chosen backend supports all required ops.
    pub fn auto() -> Result<Self, BackendError> {
        let matrix = KernelMatrix::probe();
        let compiled = Backend::compiled();

        if compiled.is_empty() {
            return Err(BackendError::NoBackendsCompiled);
        }

        // Prefer Metal on macOS, fall back to CPU.
        let active = Self::pick_best(compiled, &matrix)?;
        matrix.validate(active)?;

        Ok(Self { active, matrix })
    }

    /// Force a specific backend. Validates op support.
    pub fn with_backend(backend: Backend) -> Result<Self, BackendError> {
        let matrix = KernelMatrix::probe();
        matrix.validate(backend)?;
        Ok(Self {
            active: backend,
            matrix,
        })
    }

    /// The currently active backend.
    pub fn active(&self) -> Backend {
        self.active
    }

    /// The kernel availability matrix.
    pub fn matrix(&self) -> &KernelMatrix {
        &self.matrix
    }

    /// Pick the best backend from compiled options.
    fn pick_best(compiled: &[Backend], matrix: &KernelMatrix) -> Result<Backend, BackendError> {
        // Metal preferred when fully supported.
        #[cfg(feature = "metal")]
        {
            if compiled.contains(&Backend::Metal) && matrix.validate(Backend::Metal).is_ok() {
                return Ok(Backend::Metal);
            }
        }

        #[cfg(feature = "cpu")]
        {
            if compiled.contains(&Backend::Cpu) {
                return Ok(Backend::Cpu);
            }
        }

        // Suppress unused variable warnings when features are off.
        let _ = (compiled, matrix);
        Err(BackendError::NoBackendsCompiled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_compiled_includes_cpu() {
        let compiled = Backend::compiled();
        assert!(
            compiled.contains(&Backend::Cpu),
            "cpu feature should be enabled by default"
        );
    }

    #[test]
    fn kernel_matrix_cpu_supports_all_ops() {
        let matrix = KernelMatrix::probe();
        for &op in KernelOp::all() {
            assert!(matrix.supports(Backend::Cpu, op), "CPU should support {op}");
        }
    }

    #[test]
    fn kernel_matrix_validate_cpu_passes() {
        let matrix = KernelMatrix::probe();
        assert!(matrix.validate(Backend::Cpu).is_ok());
    }

    #[test]
    fn kernel_matrix_supported_ops_returns_all_for_cpu() {
        let matrix = KernelMatrix::probe();
        let ops = matrix.supported_ops(Backend::Cpu);
        assert_eq!(ops.len(), KernelOp::all().len());
    }

    #[test]
    fn backend_selector_auto_succeeds() {
        let selector = BackendSelector::auto().unwrap();
        // With default features (cpu), should pick CPU (Metal may not be available).
        let active = selector.active();
        assert!(
            Backend::compiled().contains(&active),
            "auto-selected backend should be a compiled backend"
        );
    }

    #[test]
    fn backend_selector_with_cpu() {
        let selector = BackendSelector::with_backend(Backend::Cpu).unwrap();
        assert_eq!(selector.active(), Backend::Cpu);
    }

    #[test]
    fn backend_display() {
        assert_eq!(format!("{}", Backend::Cpu), "cpu");
    }

    #[test]
    fn kernel_op_display() {
        assert_eq!(format!("{}", KernelOp::Attention), "attention");
        assert_eq!(format!("{}", KernelOp::RmsNorm), "rms_norm");
    }

    #[test]
    fn kernel_op_all_is_exhaustive() {
        assert_eq!(KernelOp::all().len(), 6);
    }

    #[test]
    fn backend_error_display() {
        let err = BackendError::UnsupportedOps {
            backend: Backend::Cpu,
            missing: vec![KernelOp::Attention, KernelOp::Rope],
        };
        let msg = format!("{err}");
        assert!(msg.contains("attention"));
        assert!(msg.contains("rope"));
    }
}
