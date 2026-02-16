//! MLX framework bindings for Rust.
//!
//! This module provides safe, idiomatic Rust wrappers around the MLX C API,
//! purpose-built for Apple Silicon's Unified Memory Architecture.
//!
//! # Architecture
//!
//! - [`sys`]: Raw FFI declarations (extern "C" bindings).
//! - [`array`]: Safe `Array` type with RAII lifecycle management.
//! - [`dtype`]: Data type definitions mapping to MLX's internal codes.
//! - [`ops`]: Lazy graph-building operations (arithmetic, attention, RoPE, etc.).
//! - [`metal`]: Metal backend memory management utilities.
//!
//! # Lazy Evaluation
//!
//! MLX uses lazy evaluation. Operations like `ops::add(a, b)` do NOT execute
//! immediately; they build a computation graph. Call `Array::eval()` to force
//! execution. **Never call `eval()` on the UI thread.**

// On non-macOS, FFI stubs are regular Rust functions, so `unsafe` blocks
// around their calls are technically unnecessary. Suppress the warning;
// these unsafe blocks are required on macOS with the real MLX C library.
#[allow(unused_unsafe)]
pub mod array;
pub mod dtype;
#[allow(unused_unsafe)]
pub mod metal;
#[allow(unused_unsafe)]
pub mod ops;
pub mod sys;

pub use array::Array;
pub use dtype::Dtype;
