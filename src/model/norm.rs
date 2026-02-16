//! RMSNorm (Root Mean Square Layer Normalization).
//!
//! Llama uses RMSNorm instead of LayerNorm. It normalizes by the RMS of the
//! input without centering (no bias subtraction).
//!
//! $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

use crate::mlx::ops;
use crate::mlx::Array;

/// RMSNorm layer weights and parameters.
pub struct RmsNorm {
    /// Learned scale parameter gamma, shape `[hidden_size]`.
    pub weight: Array,

    /// Small constant for numerical stability.
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Array, eps: f32) -> Self {
        RmsNorm { weight, eps }
    }

    /// Apply RMSNorm to input tensor.
    ///
    /// # Arguments
    /// - `x`: Input of shape `[batch, seq_len, hidden_size]`.
    ///
    /// # Returns
    /// Normalized tensor of the same shape.
    pub fn forward(&self, x: &Array) -> Array {
        // Compute x^2
        let x_sq = ops::multiply(x, x);

        // mean(x^2, axis=-1, keepdims=true)
        let mean_sq = ops::mean(&x_sq, &[-1], true);

        // mean + eps
        let eps = Array::from_f32(self.eps);
        let mean_sq_eps = ops::add(&mean_sq, &eps);

        // rsqrt(mean + eps)
        let inv_rms = ops::rsqrt(&mean_sq_eps);

        // x * rsqrt(mean(x^2) + eps)
        let normalized = ops::multiply(x, &inv_rms);

        // normalized * weight (broadcasting over last dim)
        ops::multiply(&normalized, &self.weight)
    }
}
