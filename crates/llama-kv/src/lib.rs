//! # llama-kv
//!
//! First-class KV cache implementation for llama.rs. Supports prefill writes,
//! single-token decode appends, paging/eviction for sliding window, and
//! memory-optimized layouts for Metal/CPU shared memory.
#![allow(unused)]
