//! Safe wrapper around the MLX array type.
//!
//! The [`Array`] type manages the lifecycle of an MLX array handle, implementing
//! reference counting via `Clone` (lightweight retain) and `Drop` (release).
//!
//! # Safety Model
//!
//! MLX arrays are reference-counted objects. `Clone` increments the refcount
//! (no data copy). Arrays may be part of a lazy computation graph; the graph
//! holds its own references, so dropping a Rust `Array` will not invalidate
//! pending computations.
//!
//! MLX arrays are thread-safe for reading (atomic refcount), so `Array`
//! implements `Send` and `Sync`.

use std::ffi::c_void;
use std::os::raw::c_int;

use crate::error::Result;
use crate::mlx::dtype::Dtype;
use crate::mlx::sys;

/// A safe handle to an MLX array.
///
/// This is a thin wrapper around an opaque pointer to `mlx::core::array`.
/// Clone is a lightweight refcount increment (no data copy).
pub struct Array {
    pub(crate) inner: sys::MlxArray,
}

// MLX arrays use atomic reference counting; safe to share across threads.
unsafe impl Send for Array {}
unsafe impl Sync for Array {}

impl Array {
    // ── Construction ────────────────────────────────────────────────────

    /// Wrap a raw MLX array pointer. The caller transfers ownership.
    ///
    /// # Safety
    /// `ptr` must be a valid, non-null MLX array handle. The caller must not
    /// call `mlx_release` on `ptr` after this call.
    pub(crate) unsafe fn from_raw(ptr: sys::MlxArray) -> Self {
        debug_assert!(!ptr.is_null(), "Array::from_raw received null pointer");
        Array { inner: ptr }
    }

    /// Create a scalar float32 array.
    pub fn from_f32(val: f32) -> Self {
        unsafe { Self::from_raw(sys::mlx_array_from_float(val)) }
    }

    /// Create a scalar int32 array.
    pub fn from_i32(val: i32) -> Self {
        unsafe { Self::from_raw(sys::mlx_array_from_int(val)) }
    }

    /// Create an array from a slice, copying the data into MLX-managed memory.
    ///
    /// This is the safe path: MLX gets its own copy, so the Rust slice can be
    /// dropped freely without risking use-after-free in lazy evaluation.
    pub fn from_slice_f32(data: &[f32], shape: &[i32]) -> Self {
        let ndim = shape.len() as c_int;
        unsafe {
            Self::from_raw(sys::mlx_array_from_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                ndim,
                Dtype::Float32.to_raw(),
                true, // copy = true for safety
            ))
        }
    }

    /// Create an array from a slice of i32, copying the data.
    pub fn from_slice_i32(data: &[i32], shape: &[i32]) -> Self {
        let ndim = shape.len() as c_int;
        unsafe {
            Self::from_raw(sys::mlx_array_from_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                ndim,
                Dtype::Int32.to_raw(),
                true,
            ))
        }
    }

    /// Create an array from a raw data pointer without copying.
    ///
    /// # Safety
    /// The caller must guarantee that `data` remains valid for the entire
    /// lifetime of this array AND any lazy computations that reference it.
    /// This is primarily used for memory-mapped safetensors weights.
    pub unsafe fn from_ptr(
        data: *const c_void,
        shape: &[i32],
        dtype: Dtype,
    ) -> Self {
        let ndim = shape.len() as c_int;
        Self::from_raw(sys::mlx_array_from_data(
            data,
            shape.as_ptr(),
            ndim,
            dtype.to_raw(),
            false, // no copy — caller must ensure lifetime
        ))
    }

    /// Create a zeros array.
    pub fn zeros(shape: &[i32], dtype: Dtype) -> Self {
        let ndim = shape.len() as c_int;
        unsafe { Self::from_raw(sys::mlx_zeros(shape.as_ptr(), ndim, dtype.to_raw())) }
    }

    /// Create a ones array.
    pub fn ones(shape: &[i32], dtype: Dtype) -> Self {
        let ndim = shape.len() as c_int;
        unsafe { Self::from_raw(sys::mlx_ones(shape.as_ptr(), ndim, dtype.to_raw())) }
    }

    /// Create an array with values from start to stop.
    pub fn arange(start: f32, stop: f32, step: f32, dtype: Dtype) -> Self {
        unsafe {
            Self::from_raw(sys::mlx_arange(start, stop, step, dtype.to_raw()))
        }
    }

    // ── Properties ──────────────────────────────────────────────────────

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        unsafe { sys::mlx_array_ndim(self.inner) as usize }
    }

    /// Shape as a vector.
    pub fn shape(&self) -> Vec<i32> {
        let ndim = self.ndim();
        if ndim == 0 {
            return vec![];
        }
        unsafe {
            let ptr = sys::mlx_array_shape(self.inner);
            std::slice::from_raw_parts(ptr, ndim).to_vec()
        }
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        unsafe { sys::mlx_array_size(self.inner) as usize }
    }

    /// Data type of the array elements.
    pub fn dtype(&self) -> Result<Dtype> {
        let code = unsafe { sys::mlx_array_dtype(self.inner) };
        Dtype::from_raw(code)
    }

    // ── Evaluation ──────────────────────────────────────────────────────

    /// Force evaluation of this array (blocks until GPU completes).
    ///
    /// **Warning**: Do not call this on the UI thread. Use the actor pattern
    /// to run inference on a dedicated thread.
    pub fn eval(&self) {
        unsafe { sys::mlx_eval(self.inner) }
    }

    /// Force evaluation of multiple arrays simultaneously.
    pub fn eval_multi(arrays: &[&Array]) {
        let ptrs: Vec<sys::MlxArray> = arrays.iter().map(|a| a.inner).collect();
        unsafe { sys::mlx_eval_multi(ptrs.as_ptr(), ptrs.len() as c_int) }
    }

    // ── Data Access ─────────────────────────────────────────────────────

    /// Get a pointer to the evaluated data. Forces evaluation if needed.
    ///
    /// The returned pointer is valid for the lifetime of this Array.
    pub fn data_ptr(&self) -> *const c_void {
        unsafe { sys::mlx_array_data_ptr(self.inner) }
    }

    /// Read the array data as a f32 slice. Forces evaluation.
    ///
    /// # Panics
    /// Panics if the dtype is not Float32.
    pub fn as_slice_f32(&self) -> &[f32] {
        let dtype = self.dtype().expect("failed to get dtype");
        assert_eq!(dtype, Dtype::Float32, "expected float32, got {dtype}");
        let ptr = self.data_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.size()) }
    }

    /// Read the scalar f32 value. Forces evaluation.
    pub fn item_f32(&self) -> f32 {
        self.as_slice_f32()[0]
    }

    /// Read the array data as an i32 slice. Forces evaluation.
    pub fn as_slice_i32(&self) -> &[i32] {
        let dtype = self.dtype().expect("failed to get dtype");
        assert_eq!(dtype, Dtype::Int32, "expected int32, got {dtype}");
        let ptr = self.data_ptr() as *const i32;
        unsafe { std::slice::from_raw_parts(ptr, self.size()) }
    }

    /// Read the scalar i32 value.
    pub fn item_i32(&self) -> i32 {
        self.as_slice_i32()[0]
    }

    /// Consume the handle and return the raw pointer without releasing.
    #[allow(dead_code)]
    pub(crate) fn into_raw(self) -> sys::MlxArray {
        let ptr = self.inner;
        std::mem::forget(self); // prevent Drop from calling mlx_release
        ptr
    }

    /// Check if the underlying pointer is null (invalid state).
    pub fn is_null(&self) -> bool {
        self.inner.is_null()
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { sys::mlx_release(self.inner) };
        }
    }
}

impl Clone for Array {
    /// Lightweight clone: increments MLX's internal reference count.
    /// No data is copied.
    fn clone(&self) -> Self {
        if !self.inner.is_null() {
            unsafe { sys::mlx_retain(self.inner) };
        }
        Array { inner: self.inner }
    }
}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.shape();
        let dtype = self.dtype().map(|d| d.name().to_string()).unwrap_or_else(|_| "?".into());
        write!(f, "Array(shape={shape:?}, dtype={dtype})")
    }
}
