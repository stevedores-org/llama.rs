//! Raw FFI bindings to the MLX C API.
//!
//! These are low-level, unsafe bindings that map directly to the MLX C interface.
//! Users should prefer the safe wrappers in [`super::array::Array`].
//!
//! On macOS with the MLX library installed, these bind to the real C symbols.
//! On other platforms, stub implementations are provided that return null
//! pointers / no-ops, allowing the code to compile and non-MLX tests to run.

use std::ffi::c_void;
use std::os::raw::{c_char, c_float, c_int};

/// Opaque handle to an MLX array object (mlx::core::array).
pub type MlxArray = *mut c_void;

/// Opaque handle to an MLX stream object.
#[allow(dead_code)]
pub type MlxStream = *mut c_void;

// ─── macOS: link against the real MLX C library ──────────────────────────────
#[cfg(target_os = "macos")]
extern "C" {
    pub fn mlx_retain(arr: MlxArray);
    pub fn mlx_release(arr: MlxArray);
    pub fn mlx_array_from_float(val: c_float) -> MlxArray;
    pub fn mlx_array_from_int(val: c_int) -> MlxArray;
    pub fn mlx_array_from_data(
        data: *const c_void, shape: *const c_int, ndim: c_int,
        dtype: c_int, copy: bool,
    ) -> MlxArray;
    pub fn mlx_zeros(shape: *const c_int, ndim: c_int, dtype: c_int) -> MlxArray;
    pub fn mlx_ones(shape: *const c_int, ndim: c_int, dtype: c_int) -> MlxArray;
    pub fn mlx_full(shape: *const c_int, val: MlxArray, dtype: c_int) -> MlxArray;
    pub fn mlx_arange(start: c_float, stop: c_float, step: c_float, dtype: c_int) -> MlxArray;
    pub fn mlx_array_ndim(arr: MlxArray) -> c_int;
    pub fn mlx_array_shape(arr: MlxArray) -> *const c_int;
    pub fn mlx_array_size(arr: MlxArray) -> c_int;
    pub fn mlx_array_dtype(arr: MlxArray) -> c_int;
    pub fn mlx_array_data_ptr(arr: MlxArray) -> *const c_void;
    pub fn mlx_eval(arr: MlxArray);
    pub fn mlx_eval_multi(arrs: *const MlxArray, count: c_int);
    pub fn mlx_add(a: MlxArray, b: MlxArray) -> MlxArray;
    pub fn mlx_subtract(a: MlxArray, b: MlxArray) -> MlxArray;
    pub fn mlx_multiply(a: MlxArray, b: MlxArray) -> MlxArray;
    pub fn mlx_divide(a: MlxArray, b: MlxArray) -> MlxArray;
    pub fn mlx_negative(a: MlxArray) -> MlxArray;
    pub fn mlx_sqrt(a: MlxArray) -> MlxArray;
    pub fn mlx_rsqrt(a: MlxArray) -> MlxArray;
    pub fn mlx_exp(a: MlxArray) -> MlxArray;
    pub fn mlx_log(a: MlxArray) -> MlxArray;
    pub fn mlx_abs(a: MlxArray) -> MlxArray;
    pub fn mlx_power(a: MlxArray, b: MlxArray) -> MlxArray;
    pub fn mlx_maximum(a: MlxArray, b: MlxArray) -> MlxArray;
    pub fn mlx_minimum(a: MlxArray, b: MlxArray) -> MlxArray;
    pub fn mlx_sum(a: MlxArray, axes: *const c_int, n: c_int, kd: bool) -> MlxArray;
    pub fn mlx_mean(a: MlxArray, axes: *const c_int, n: c_int, kd: bool) -> MlxArray;
    pub fn mlx_max(a: MlxArray, axes: *const c_int, n: c_int, kd: bool) -> MlxArray;
    pub fn mlx_argmax(a: MlxArray, axis: c_int, keepdims: bool) -> MlxArray;
    pub fn mlx_argpartition(a: MlxArray, kth: c_int, axis: c_int) -> MlxArray;
    pub fn mlx_sort(a: MlxArray, axis: c_int) -> MlxArray;
    pub fn mlx_cumsum(a: MlxArray, axis: c_int, reverse: bool, inclusive: bool) -> MlxArray;
    pub fn mlx_matmul(a: MlxArray, b: MlxArray) -> MlxArray;
    pub fn mlx_quantized_matmul(
        x: MlxArray, w: MlxArray, scales: MlxArray, biases: MlxArray,
        transpose: bool, group_size: c_int, bits: c_int,
    ) -> MlxArray;
    pub fn mlx_reshape(a: MlxArray, shape: *const c_int, ndim: c_int) -> MlxArray;
    pub fn mlx_transpose(a: MlxArray, axes: *const c_int, naxes: c_int) -> MlxArray;
    pub fn mlx_expand_dims(a: MlxArray, axis: c_int) -> MlxArray;
    pub fn mlx_squeeze(a: MlxArray, axis: c_int) -> MlxArray;
    pub fn mlx_concatenate(arrays: *const MlxArray, count: c_int, axis: c_int) -> MlxArray;
    pub fn mlx_split(a: MlxArray, num_splits: c_int, axis: c_int) -> *mut MlxArray;
    pub fn mlx_slice(
        a: MlxArray, start: *const c_int, stop: *const c_int,
        strides: *const c_int, ndim: c_int,
    ) -> MlxArray;
    pub fn mlx_slice_update(
        a: MlxArray, update: MlxArray, start: *const c_int,
        stop: *const c_int, strides: *const c_int, ndim: c_int,
    ) -> MlxArray;
    pub fn mlx_broadcast_to(a: MlxArray, shape: *const c_int, ndim: c_int) -> MlxArray;
    pub fn mlx_repeat(a: MlxArray, repeats: c_int, axis: c_int) -> MlxArray;
    pub fn mlx_softmax(a: MlxArray, axis: c_int) -> MlxArray;
    pub fn mlx_sigmoid(a: MlxArray) -> MlxArray;
    pub fn mlx_silu(a: MlxArray) -> MlxArray;
    pub fn mlx_gelu(a: MlxArray) -> MlxArray;
    pub fn mlx_scaled_dot_product_attention(
        queries: MlxArray, keys: MlxArray, values: MlxArray,
        scale: c_float, mask: MlxArray,
    ) -> MlxArray;
    pub fn mlx_fast_rope(
        x: MlxArray, dims: c_int, traditional: bool,
        base: c_float, scale: c_float, offset: c_int,
    ) -> MlxArray;
    pub fn mlx_astype(a: MlxArray, dtype: c_int) -> MlxArray;
    pub fn mlx_less(a: MlxArray, b: MlxArray) -> MlxArray;
    pub fn mlx_greater(a: MlxArray, b: MlxArray) -> MlxArray;
    pub fn mlx_equal(a: MlxArray, b: MlxArray) -> MlxArray;
    pub fn mlx_where(condition: MlxArray, x: MlxArray, y: MlxArray) -> MlxArray;
    pub fn mlx_random_categorical(logits: MlxArray, axis: c_int) -> MlxArray;
    pub fn mlx_metal_clear_cache();
    pub fn mlx_metal_get_active_memory() -> usize;
    pub fn mlx_metal_get_peak_memory() -> usize;
    pub fn mlx_metal_set_memory_limit(limit: usize, relaxed: bool) -> usize;
    pub fn mlx_metal_set_cache_limit(limit: usize) -> usize;
    pub fn mlx_array_tostring(arr: MlxArray) -> *mut c_char;
    pub fn mlx_free_string(s: *mut c_char);
}

// ─── Non-macOS: stub implementations ─────────────────────────────────────────
//
// These stubs return null pointers and perform no operations.  They allow the
// crate to compile and link on Linux/Windows so that non-MLX components
// (config parsing, safetensors loading, sampling config, tokenizer, etc.)
// can be tested without the Apple-only MLX library.

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc, unused_variables)]
pub mod stubs {
    use super::*;
    use std::ptr;

    const NULL: MlxArray = ptr::null_mut();

    #[no_mangle] pub extern "C" fn mlx_retain(_: MlxArray) {}
    #[no_mangle] pub extern "C" fn mlx_release(_: MlxArray) {}
    #[no_mangle] pub extern "C" fn mlx_array_from_float(_: c_float) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_array_from_int(_: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_array_from_data(
        _: *const c_void, _: *const c_int, _: c_int, _: c_int, _: bool,
    ) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_zeros(_: *const c_int, _: c_int, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_ones(_: *const c_int, _: c_int, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_full(_: *const c_int, _: MlxArray, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_arange(_: c_float, _: c_float, _: c_float, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_array_ndim(_: MlxArray) -> c_int { 0 }
    #[no_mangle] pub extern "C" fn mlx_array_shape(_: MlxArray) -> *const c_int { ptr::null() }
    #[no_mangle] pub extern "C" fn mlx_array_size(_: MlxArray) -> c_int { 0 }
    #[no_mangle] pub extern "C" fn mlx_array_dtype(_: MlxArray) -> c_int { 9 }
    #[no_mangle] pub extern "C" fn mlx_array_data_ptr(_: MlxArray) -> *const c_void { ptr::null() }
    #[no_mangle] pub extern "C" fn mlx_eval(_: MlxArray) {}
    #[no_mangle] pub extern "C" fn mlx_eval_multi(_: *const MlxArray, _: c_int) {}
    #[no_mangle] pub extern "C" fn mlx_add(_: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_subtract(_: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_multiply(_: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_divide(_: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_negative(_: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_sqrt(_: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_rsqrt(_: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_exp(_: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_log(_: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_abs(_: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_power(_: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_maximum(_: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_minimum(_: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_sum(_: MlxArray, _: *const c_int, _: c_int, _: bool) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_mean(_: MlxArray, _: *const c_int, _: c_int, _: bool) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_max(_: MlxArray, _: *const c_int, _: c_int, _: bool) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_argmax(_: MlxArray, _: c_int, _: bool) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_argpartition(_: MlxArray, _: c_int, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_sort(_: MlxArray, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_cumsum(_: MlxArray, _: c_int, _: bool, _: bool) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_matmul(_: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_quantized_matmul(
        _: MlxArray, _: MlxArray, _: MlxArray, _: MlxArray,
        _: bool, _: c_int, _: c_int,
    ) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_reshape(_: MlxArray, _: *const c_int, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_transpose(_: MlxArray, _: *const c_int, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_expand_dims(_: MlxArray, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_squeeze(_: MlxArray, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_concatenate(_: *const MlxArray, _: c_int, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_split(_: MlxArray, _: c_int, _: c_int) -> *mut MlxArray { ptr::null_mut() }
    #[no_mangle] pub extern "C" fn mlx_slice(
        _: MlxArray, _: *const c_int, _: *const c_int, _: *const c_int, _: c_int,
    ) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_slice_update(
        _: MlxArray, _: MlxArray, _: *const c_int, _: *const c_int, _: *const c_int, _: c_int,
    ) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_broadcast_to(_: MlxArray, _: *const c_int, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_repeat(_: MlxArray, _: c_int, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_softmax(_: MlxArray, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_sigmoid(_: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_silu(_: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_gelu(_: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_scaled_dot_product_attention(
        _: MlxArray, _: MlxArray, _: MlxArray, _: c_float, _: MlxArray,
    ) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_fast_rope(
        _: MlxArray, _: c_int, _: bool, _: c_float, _: c_float, _: c_int,
    ) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_astype(_: MlxArray, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_less(_: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_greater(_: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_equal(_: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_where(_: MlxArray, _: MlxArray, _: MlxArray) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_random_categorical(_: MlxArray, _: c_int) -> MlxArray { NULL }
    #[no_mangle] pub extern "C" fn mlx_metal_clear_cache() {}
    #[no_mangle] pub extern "C" fn mlx_metal_get_active_memory() -> usize { 0 }
    #[no_mangle] pub extern "C" fn mlx_metal_get_peak_memory() -> usize { 0 }
    #[no_mangle] pub extern "C" fn mlx_metal_set_memory_limit(_: usize, _: bool) -> usize { 0 }
    #[no_mangle] pub extern "C" fn mlx_metal_set_cache_limit(_: usize) -> usize { 0 }
    #[no_mangle] pub extern "C" fn mlx_array_tostring(_: MlxArray) -> *mut c_char { ptr::null_mut() }
    #[no_mangle] pub extern "C" fn mlx_free_string(_: *mut c_char) {}
}

#[cfg(not(target_os = "macos"))]
pub use stubs::*;
