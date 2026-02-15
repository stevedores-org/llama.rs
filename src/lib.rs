//! Rust bindings and utilities for llama.cpp
//!
//! This crate provides Rust bindings to the llama.cpp library for efficient LLM inference.

pub mod model;
pub mod session;

pub use model::Model;
pub use session::Session;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
