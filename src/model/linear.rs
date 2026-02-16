//! Linear and quantized linear layers.
//!
//! Standard linear layers perform `y = x @ W^T + b`.
//! Quantized linear layers use packed integer weights with per-group scales
//! and biases, decompressed on-the-fly by the Metal kernel.

use crate::mlx::ops;
use crate::mlx::Array;

/// Standard (full-precision) linear layer.
pub struct Linear {
    /// Weight matrix, shape `[out_features, in_features]`.
    pub weight: Array,

    /// Optional bias, shape `[out_features]`.
    pub bias: Option<Array>,
}

impl Linear {
    pub fn new(weight: Array, bias: Option<Array>) -> Self {
        Linear { weight, bias }
    }

    /// Forward pass: `y = x @ W^T [+ b]`.
    pub fn forward(&self, x: &Array) -> Array {
        // Transpose weight for matmul: [out, in] -> [in, out]
        let wt = ops::transpose(&self.weight, &[1, 0]);
        let y = ops::matmul(x, &wt);
        match &self.bias {
            Some(b) => ops::add(&y, b),
            None => y,
        }
    }
}

/// Quantized linear layer using MLX's group-wise affine quantization.
///
/// The weight tensor stores packed integer values (e.g., 8 x 4-bit weights
/// per uint32). Decompression happens in the Metal kernel during matmul,
/// avoiding full materialization of the fp16 weight matrix.
pub struct QuantizedLinear {
    /// Packed quantized weights.
    pub weight: Array,

    /// Per-group scale factors, shape `[out_features, in_features / group_size]`.
    pub scales: Array,

    /// Per-group bias/zero-point, shape `[out_features, in_features / group_size]`.
    pub biases: Array,

    /// Number of elements per quantization group (e.g., 32 or 64).
    pub group_size: i32,

    /// Quantization bit-depth (e.g., 4 or 8).
    pub bits: i32,
}

impl QuantizedLinear {
    pub fn new(
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    ) -> Self {
        QuantizedLinear {
            weight,
            scales,
            biases,
            group_size,
            bits,
        }
    }

    /// Forward pass using the quantized matmul kernel.
    ///
    /// The Metal kernel reads packed weights, decompresses in-register,
    /// multiplies, and discards the decompressed values. This avoids
    /// materializing the full fp16 weight matrix in memory.
    pub fn forward(&self, x: &Array) -> Array {
        ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            true, // transpose
            self.group_size,
            self.bits,
        )
    }
}

/// A linear layer that can be either full-precision or quantized.
pub enum AnyLinear {
    Full(Linear),
    Quantized(QuantizedLinear),
}

impl AnyLinear {
    pub fn forward(&self, x: &Array) -> Array {
        match self {
            AnyLinear::Full(l) => l.forward(x),
            AnyLinear::Quantized(q) => q.forward(x),
        }
    }
}
