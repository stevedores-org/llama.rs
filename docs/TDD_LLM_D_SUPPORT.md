# TDD Plan: llm-d Distributed Inference Support

> Add llm-d (Kubernetes-native distributed LLM inference) support to llama.rs

## Background

[llm-d](https://llm-d.ai/) is a Kubernetes-native distributed inference serving stack with founding contributors including Red Hat, IBM Research, Google Cloud, NVIDIA, and CoreWeave. It provides three core capabilities:

1. **Disaggregated prefill/decode serving** — separate pod workloads for compute-bound prefill and memory-bandwidth-bound decode phases
2. **Intelligent inference scheduling** — vLLM-aware load balancing via the Inference Gateway (IGW) with prefix cache awareness
3. **Multi-tier KV cache management** — KV cache transfer between instances over fast interconnects (RDMA, NIXL)

### Why llama.rs is Well-Positioned

The `LlamaEngine` trait already separates `prefill()` and `decode()` as distinct operations — this maps directly to llm-d's disaggregated serving model. The OpenAI-compatible server, KV cache with per-layer granularity, and health endpoint are also aligned.

### Key Gaps

| Gap | Current State | Required |
|-----|--------------|----------|
| KV cache serialization | In-process only (`llama-kv`) | Serialize/deserialize to bytes for network transfer |
| Operating modes | Server always does prefill + decode | Prefill-only and decode-only modes |
| Scheduling telemetry | No metrics endpoint | Expose queue depth, KV utilization, prefix cache hits |
| Header-aware routing | Not implemented | Handle `x-prefiller-url`, `x-session-token` |
| KV transfer protocol | None | gRPC or sidecar-compatible transfer |
| Health probes | Basic `/health` | Kubernetes liveness/readiness with model-loaded state |
| Prefix cache | No prefix matching | Report prefix cache hit probability to scheduler |

## Phased Implementation

### Phase 1: KV Cache Serialization (`llama-kv`)

**Goal:** Enable KV cache data to be serialized to bytes and deserialized back, preserving correctness.

**New trait in `crates/llama-kv/src/lib.rs`:**

```rust
pub trait KvTransfer: Send + Sync {
    fn serialize_session(&self, session: &SessionKVCache) -> Result<Vec<u8>>;
    fn deserialize_session(&self, data: &[u8]) -> Result<SessionKVCache>;
    fn serialize_prefix(&self, session: &SessionKVCache, token_count: usize) -> Result<Vec<u8>>;
}
```

**TDD sequence:**

1. `test_roundtrip_empty_cache` — serialize empty `SessionKVCache`, deserialize, assert equality
2. `test_roundtrip_single_layer` — write known f32 values to one layer, roundtrip, compare bitwise
3. `test_roundtrip_multi_layer_all_layouts` — test BySequence, ByHead, Transposed layouts
4. `test_prefix_serialization` — serialize first N tokens of a populated cache, verify truncation
5. `test_deserialize_corrupt_data` — bad bytes return `Err`, not panic
6. `test_serialize_large_cache_performance` — benchmark: 32 layers, 4096 seq_len, 128 head_dim should serialize in < 100ms

**Implementation:** Use a simple binary format: header (version, n_layers, layout, seq_len, n_heads, head_dim) + flat f32 data. No compression in v1.

### Phase 2: Disaggregated Server Modes (`llama-server`)

**Goal:** The server can run as prefill-only, decode-only, or full (default).

**Config addition to `ServerConfig`:**

```rust
pub enum ServingMode {
    Full,       // Both prefill and decode (default, current behavior)
    Prefill,    // Only accept prefill requests, return KV cache reference
    Decode,     // Only accept decode requests, expect KV cache input
}
```

**TDD sequence:**

1. `test_prefill_mode_returns_kv_reference` — POST `/v1/chat/completions` in prefill mode returns a JSON response with `kv_cache_id` and `token_count` instead of generated text
2. `test_prefill_mode_rejects_decode_header` — request with `x-session-token` to a prefill-only server returns 400
3. `test_decode_mode_accepts_kv_transfer` — POST with `x-kv-cache-id` header triggers KV cache load then decode loop
4. `test_decode_mode_rejects_without_kv` — request without KV cache reference returns 400
5. `test_full_mode_unchanged` — default mode behaves exactly like current server (regression)
6. `test_prefill_mode_health_reports_mode` — `/health` response includes `"mode": "prefill"`
7. `test_x_prefiller_url_header_forwarding` — decode instance reads `x-prefiller-url` from request header

**Implementation:**
- Add `--mode prefill|decode|full` CLI flag
- In prefill mode: run `engine.prefill()`, serialize KV cache, store in local buffer with UUID key, return `{kv_cache_id, token_count, prefill_time_ms}`
- In decode mode: accept `kv_cache_id`, deserialize KV cache into session, run decode loop as normal
- The KV transfer between instances is handled by the llm-d sidecar (Phase 4)

### Phase 3: Scheduling Telemetry (`llama-server`)

**Goal:** Expose metrics that llm-d's Endpoint Picker Protocol (EPP) consumes for intelligent routing.

**New endpoint: `GET /metrics`**

```json
{
  "active_requests": 3,
  "queue_depth": 7,
  "kv_cache_utilization": 0.73,
  "prefix_cache_hit_rate": 0.45,
  "serving_mode": "decode",
  "max_concurrent_sessions": 8,
  "model_loaded": true
}
```

**TDD sequence:**

1. `test_metrics_endpoint_returns_200` — GET `/metrics` returns 200 with correct content-type
2. `test_metrics_active_requests_increments` — during an in-flight request, `active_requests > 0`
3. `test_metrics_queue_depth_under_load` — with semaphore full, queued requests increment `queue_depth`
4. `test_metrics_kv_utilization_reflects_sessions` — after creating sessions, utilization > 0
5. `test_metrics_reports_serving_mode` — matches configured `ServingMode`
6. `test_metrics_model_loaded_false_before_init` — before model load, reports `false`

**Implementation:**
- Add `MetricsCollector` to `AppState` with atomic counters
- `SessionManager` already tracks active sessions; expose via metrics
- KV utilization = total KV cache bytes / configured max

### Phase 4: KV Transfer Protocol (new crate `llama-kv-transfer`)

**Goal:** Enable KV cache transfer between prefill and decode instances over the network.

**New crate: `crates/llama-kv-transfer/`**

```rust
pub trait KvTransport: Send + Sync {
    async fn send(&self, cache_id: &str, data: &[u8]) -> Result<()>;
    async fn recv(&self, cache_id: &str) -> Result<Vec<u8>>;
    async fn exists(&self, cache_id: &str) -> Result<bool>;
}
```

**TDD sequence:**

1. `test_in_memory_transport_roundtrip` — send bytes, recv same bytes
2. `test_in_memory_transport_not_found` — recv unknown ID returns Err
3. `test_in_memory_transport_overwrite` — send same ID twice, recv gets latest
4. `test_tcp_transport_roundtrip` — start local TCP listener, send/recv across sockets
5. `test_tcp_transport_timeout` — connection to unreachable host times out
6. `test_sidecar_transport_http` — mock HTTP sidecar endpoint, verify POST/GET pattern

**Implementations:**
- `InMemoryTransport` — for testing and single-node development
- `TcpTransport` — direct TCP for simple deployments
- `SidecarTransport` — HTTP client compatible with llm-d's NIXL sidecar protocol

### Phase 5: Kubernetes Health Probes (`llama-server`)

**Goal:** Enhanced health endpoints for Kubernetes liveness and readiness.

**TDD sequence:**

1. `test_liveness_always_200` — GET `/healthz` returns 200 if process is alive
2. `test_readiness_503_before_model_load` — GET `/readyz` returns 503 before model is loaded
3. `test_readiness_200_after_model_load` — returns 200 after successful model load
4. `test_readiness_503_when_overloaded` — returns 503 when queue exceeds threshold
5. `test_startup_probe` — GET `/startupz` returns 200 only after full initialization

**Implementation:**
- `/healthz` — liveness, always 200 if server is running
- `/readyz` — readiness, checks model loaded + queue not full
- `/startupz` — startup, checks one-time initialization complete

### Phase 6: Prefix Cache Awareness (`llama-kv`)

**Goal:** Enable prefix matching so the scheduler can route requests to instances that already have relevant KV cache state.

**TDD sequence:**

1. `test_prefix_match_exact` — tokens [1,2,3] match cached [1,2,3] → 100%
2. `test_prefix_match_partial` — tokens [1,2,3,4] match cached [1,2,3] → 75%
3. `test_prefix_match_none` — tokens [5,6,7] match cached [1,2,3] → 0%
4. `test_prefix_match_empty_cache` — any tokens against empty cache → 0%
5. `test_prefix_index_multiple_sessions` — index tracks prefixes across sessions
6. `test_prefix_index_eviction` — evicted sessions are removed from index

**Implementation:**
- Add `PrefixIndex` to `SessionManager` — maps token prefix hashes to session IDs
- Expose prefix match probability via `/metrics` endpoint
- llm-d scheduler uses this to route requests to instances with cached prefixes

## Crate Dependency Graph

```
llama-kv-transfer (new)
    └── llama-kv (serialize/deserialize)

llama-server
    ├── llama-engine (trait)
    ├── llama-kv (prefix index)
    ├── llama-kv-transfer (transport)
    └── llama-sampling
```

## Milestone Criteria

| Phase | Done When |
|-------|-----------|
| 1 | KV cache roundtrip tests pass for all layouts, benchmarks under threshold |
| 2 | Server runs in prefill-only and decode-only modes, passes all mode tests |
| 3 | `/metrics` endpoint returns accurate real-time telemetry |
| 4 | KV cache transfers between two server instances via TCP transport |
| 5 | Kubernetes probes respond correctly to lifecycle states |
| 6 | Prefix cache hit rate is reported and scheduler can use it for routing |
| E2E | Two-pod (prefill + decode) deployment on kind/minikube generates correct text |

## References

- [llm-d Architecture](https://llm-d.ai/docs/architecture)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [vLLM Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [vLLM KVConnector API (RFC #10818)](https://github.com/vllm-project/vllm/issues/10818)
- [llama.rs Architecture](./ARCHITECTURE.md)
- [llama.rs Test Strategy](./TEST_STRATEGY.md)
