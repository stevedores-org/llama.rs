//! # llama-runtime
//!
//! Runtime backend selection and execution for llama.rs. Integrates with oxidizedMLX
//! for CPU/Metal backends, manages tensor allocation, kernel availability matrix,
//! and telemetry hooks (TTFT, tok/s, memory).
#![allow(unused)]
