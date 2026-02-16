//! High-level MLX operations exposed as safe Rust functions.
//!
//! Each function constructs a node in the lazy computation graph.
//! No computation occurs until [`Array::eval`] is called.

use std::os::raw::c_int;

use crate::mlx::array::Array;
use crate::mlx::dtype::Dtype;
use crate::mlx::sys;

// ── Arithmetic ──────────────────────────────────────────────────────────

pub fn add(a: &Array, b: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_add(a.inner, b.inner)) }
}

pub fn subtract(a: &Array, b: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_subtract(a.inner, b.inner)) }
}

pub fn multiply(a: &Array, b: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_multiply(a.inner, b.inner)) }
}

pub fn divide(a: &Array, b: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_divide(a.inner, b.inner)) }
}

pub fn negative(a: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_negative(a.inner)) }
}

pub fn sqrt(a: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_sqrt(a.inner)) }
}

pub fn rsqrt(a: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_rsqrt(a.inner)) }
}

pub fn exp(a: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_exp(a.inner)) }
}

pub fn log(a: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_log(a.inner)) }
}

pub fn abs(a: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_abs(a.inner)) }
}

pub fn power(a: &Array, b: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_power(a.inner, b.inner)) }
}

pub fn maximum(a: &Array, b: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_maximum(a.inner, b.inner)) }
}

pub fn minimum(a: &Array, b: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_minimum(a.inner, b.inner)) }
}

// ── Reductions ──────────────────────────────────────────────────────────

pub fn sum(a: &Array, axes: &[i32], keepdims: bool) -> Array {
    unsafe {
        Array::from_raw(sys::mlx_sum(
            a.inner,
            axes.as_ptr(),
            axes.len() as c_int,
            keepdims,
        ))
    }
}

pub fn mean(a: &Array, axes: &[i32], keepdims: bool) -> Array {
    unsafe {
        Array::from_raw(sys::mlx_mean(
            a.inner,
            axes.as_ptr(),
            axes.len() as c_int,
            keepdims,
        ))
    }
}

pub fn max(a: &Array, axes: &[i32], keepdims: bool) -> Array {
    unsafe {
        Array::from_raw(sys::mlx_max(
            a.inner,
            axes.as_ptr(),
            axes.len() as c_int,
            keepdims,
        ))
    }
}

pub fn argmax(a: &Array, axis: i32, keepdims: bool) -> Array {
    unsafe { Array::from_raw(sys::mlx_argmax(a.inner, axis, keepdims)) }
}

pub fn argpartition(a: &Array, kth: i32, axis: i32) -> Array {
    unsafe { Array::from_raw(sys::mlx_argpartition(a.inner, kth, axis)) }
}

pub fn sort(a: &Array, axis: i32) -> Array {
    unsafe { Array::from_raw(sys::mlx_sort(a.inner, axis)) }
}

pub fn cumsum(a: &Array, axis: i32, reverse: bool, inclusive: bool) -> Array {
    unsafe { Array::from_raw(sys::mlx_cumsum(a.inner, axis, reverse, inclusive)) }
}

// ── Linear Algebra ──────────────────────────────────────────────────────

pub fn matmul(a: &Array, b: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_matmul(a.inner, b.inner)) }
}

/// Quantized matrix multiplication.
///
/// Multiplies `x` by quantized weight `w` using the given scales and biases.
/// This invokes an optimized Metal kernel that decompresses weights on-the-fly.
pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    transpose: bool,
    group_size: i32,
    bits: i32,
) -> Array {
    unsafe {
        Array::from_raw(sys::mlx_quantized_matmul(
            x.inner,
            w.inner,
            scales.inner,
            biases.inner,
            transpose,
            group_size,
            bits,
        ))
    }
}

// ── Shape Manipulation ──────────────────────────────────────────────────

pub fn reshape(a: &Array, shape: &[i32]) -> Array {
    unsafe {
        Array::from_raw(sys::mlx_reshape(
            a.inner,
            shape.as_ptr(),
            shape.len() as c_int,
        ))
    }
}

pub fn transpose(a: &Array, axes: &[i32]) -> Array {
    unsafe {
        Array::from_raw(sys::mlx_transpose(
            a.inner,
            axes.as_ptr(),
            axes.len() as c_int,
        ))
    }
}

pub fn expand_dims(a: &Array, axis: i32) -> Array {
    unsafe { Array::from_raw(sys::mlx_expand_dims(a.inner, axis)) }
}

pub fn squeeze(a: &Array, axis: i32) -> Array {
    unsafe { Array::from_raw(sys::mlx_squeeze(a.inner, axis)) }
}

pub fn concatenate(arrays: &[&Array], axis: i32) -> Array {
    let ptrs: Vec<sys::MlxArray> = arrays.iter().map(|a| a.inner).collect();
    unsafe {
        Array::from_raw(sys::mlx_concatenate(
            ptrs.as_ptr(),
            ptrs.len() as c_int,
            axis,
        ))
    }
}

pub fn slice(a: &Array, start: &[i32], stop: &[i32], strides: &[i32]) -> Array {
    let ndim = start.len() as c_int;
    unsafe {
        Array::from_raw(sys::mlx_slice(
            a.inner,
            start.as_ptr(),
            stop.as_ptr(),
            strides.as_ptr(),
            ndim,
        ))
    }
}

/// Update a slice of the array in-place (returns new array in the graph).
pub fn slice_update(
    a: &Array,
    update: &Array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
) -> Array {
    let ndim = start.len() as c_int;
    unsafe {
        Array::from_raw(sys::mlx_slice_update(
            a.inner,
            update.inner,
            start.as_ptr(),
            stop.as_ptr(),
            strides.as_ptr(),
            ndim,
        ))
    }
}

pub fn broadcast_to(a: &Array, shape: &[i32]) -> Array {
    unsafe {
        Array::from_raw(sys::mlx_broadcast_to(
            a.inner,
            shape.as_ptr(),
            shape.len() as c_int,
        ))
    }
}

pub fn repeat(a: &Array, repeats: i32, axis: i32) -> Array {
    unsafe { Array::from_raw(sys::mlx_repeat(a.inner, repeats, axis)) }
}

// ── Activation Functions ────────────────────────────────────────────────

pub fn softmax(a: &Array, axis: i32) -> Array {
    unsafe { Array::from_raw(sys::mlx_softmax(a.inner, axis)) }
}

pub fn sigmoid(a: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_sigmoid(a.inner)) }
}

pub fn silu(a: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_silu(a.inner)) }
}

pub fn gelu(a: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_gelu(a.inner)) }
}

// ── Attention ───────────────────────────────────────────────────────────

/// Scaled dot-product attention using the optimized FlashAttention kernel
/// on Metal.
///
/// Arguments:
/// - `queries`:  `[batch, n_heads, seq_len, head_dim]`
/// - `keys`:     `[batch, n_kv_heads, kv_len, head_dim]`
/// - `values`:   `[batch, n_kv_heads, kv_len, head_dim]`
/// - `scale`:    typically `1.0 / sqrt(head_dim)`
/// - `mask`:     optional attention mask, or `None`
pub fn scaled_dot_product_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: Option<&Array>,
) -> Array {
    let mask_ptr = mask.map_or(std::ptr::null_mut(), |m| m.inner);
    unsafe {
        Array::from_raw(sys::mlx_scaled_dot_product_attention(
            queries.inner,
            keys.inner,
            values.inner,
            scale,
            mask_ptr,
        ))
    }
}

// ── RoPE ────────────────────────────────────────────────────────────────

/// Apply Rotary Position Embeddings using the optimized Metal kernel.
///
/// Arguments:
/// - `x`:           input tensor `[batch, seq_len, n_heads, head_dim]`
/// - `dims`:        number of dimensions to rotate (typically head_dim)
/// - `traditional`: use the traditional (interleaved) RoPE layout
/// - `base`:        base frequency (default 10000.0 for Llama)
/// - `scale`:       scaling factor (1.0 for standard RoPE)
/// - `offset`:      position offset — **critical** for decode phase
pub fn fast_rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: i32,
) -> Array {
    unsafe {
        Array::from_raw(sys::mlx_fast_rope(
            x.inner, dims, traditional, base, scale, offset,
        ))
    }
}

// ── Type Casting ────────────────────────────────────────────────────────

pub fn astype(a: &Array, dtype: Dtype) -> Array {
    unsafe { Array::from_raw(sys::mlx_astype(a.inner, dtype.to_raw())) }
}

// ── Comparison ──────────────────────────────────────────────────────────

pub fn less(a: &Array, b: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_less(a.inner, b.inner)) }
}

pub fn greater(a: &Array, b: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_greater(a.inner, b.inner)) }
}

pub fn equal(a: &Array, b: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_equal(a.inner, b.inner)) }
}

pub fn where_cond(condition: &Array, x: &Array, y: &Array) -> Array {
    unsafe { Array::from_raw(sys::mlx_where(condition.inner, x.inner, y.inner)) }
}

// ── RNG ─────────────────────────────────────────────────────────────────

/// Sample from a categorical distribution defined by logits.
pub fn random_categorical(logits: &Array, axis: i32) -> Array {
    unsafe { Array::from_raw(sys::mlx_random_categorical(logits.inner, axis)) }
}
