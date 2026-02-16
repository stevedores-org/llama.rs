//! Metal backend utilities for memory management and diagnostics.
//!
//! These functions map directly to `mlx::core::metal::*` and are used to
//! control the GPU memory pool. In a long-running application like llama.app,
//! periodic cache clearing prevents the OS from killing the process under
//! memory pressure.

use crate::mlx::sys;

/// Clear the Metal memory cache.
///
/// Call this when the application is backgrounded, after a long generation
/// completes, or whenever memory pressure is detected.
pub fn clear_cache() {
    unsafe { sys::mlx_metal_clear_cache() }
}

/// Get current active GPU memory usage in bytes.
pub fn active_memory() -> usize {
    unsafe { sys::mlx_metal_get_active_memory() }
}

/// Get peak GPU memory usage in bytes since process start.
pub fn peak_memory() -> usize {
    unsafe { sys::mlx_metal_get_peak_memory() }
}

/// Set the memory limit for the Metal allocator.
///
/// - `limit`: maximum bytes the allocator may use.
/// - `relaxed`: if true, allows exceeding the limit when necessary.
///
/// Returns the previous limit.
pub fn set_memory_limit(limit: usize, relaxed: bool) -> usize {
    unsafe { sys::mlx_metal_set_memory_limit(limit, relaxed) }
}

/// Set the cache limit for the Metal allocator.
///
/// Buffers beyond this size will be freed eagerly instead of cached.
/// Returns the previous limit.
pub fn set_cache_limit(limit: usize) -> usize {
    unsafe { sys::mlx_metal_set_cache_limit(limit) }
}

/// Human-readable memory report.
pub fn memory_report() -> String {
    let active = active_memory();
    let peak = peak_memory();
    format!(
        "Metal memory: {:.1} MB active, {:.1} MB peak",
        active as f64 / (1024.0 * 1024.0),
        peak as f64 / (1024.0 * 1024.0),
    )
}
