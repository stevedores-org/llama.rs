import type { FC } from "hono/jsx";
import { Layout } from "../components/Layout";

const BASE = "/llama-rs";

export const IndexPage: FC = () => (
  <Layout title="Introduction" activePath="/">
    <div class="hero">
      <h1>llama.rs</h1>
      <p class="lead">
        A modular Rust inference runtime for Llama-family models. Built on{" "}
        <a href="https://github.com/stevedores-org/oxidizedMLX">oxidizedMLX</a> for
        Metal/CPU acceleration, with integrations for{" "}
        <a href="https://github.com/stevedores-org/oxidizedRAG">oxidizedRAG</a> and{" "}
        <a href="https://github.com/stevedores-org/oxidizedgraph">oxidizedgraph</a>.
      </p>
      <div class="hero-actions">
        <a href={`${BASE}/getting-started`} class="btn btn-primary">
          Get Started
        </a>
        <a
          href="https://github.com/stevedores-org/llama.rs"
          class="btn btn-ghost"
        >
          GitHub
        </a>
        <a href="https://crates.io/crates/llama-engine" class="btn btn-ghost">
          crates.io
        </a>
      </div>
    </div>

    <h2>Design Philosophy</h2>
    <p>
      llama.rs uses a <strong>"narrow waist"</strong> design: the{" "}
      <code>llama-engine</code> crate defines the core <code>LlamaEngine</code>{" "}
      trait that all other crates depend on. Implementations can swap CPU/Metal/FFI
      backends without changing application code.
    </p>

    <div class="callout">
      <span class="callout-icon">&#x26A1;</span>
      <div>
        <strong>Performance is a feature.</strong> Zero-copy IO via mmap, bounded
        memory via explicit KV cache policies, and cooperative cancellation from
        day one.
      </div>
    </div>

    <div class="arch-diagram">
{`                    ┌──────────────────────┐
                    │    Application       │
                    │  (llama-cli, server)  │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │    `}<span class="highlight">llama-engine</span>{`       │
                    │   (narrow waist)     │
                    └──┬───────┬───────┬───┘
                       │       │       │
          ┌────────────┘       │       └────────────┐
          ▼                    ▼                    ▼
  ┌───────────────┐  ┌────────────────┐  ┌───────────────┐
  │ llama-models  │  │ llama-runtime  │  │  llama-kv     │
  │  (RMSNorm,    │  │  (backend      │  │  (KV cache    │
  │   RoPE, MLP)  │  │   selection)   │  │   management) │
  └───────────────┘  └────────────────┘  └───────────────┘
          │                    │
  ┌───────────────┐  ┌────────────────┐
  │llama-tokenizer│  │ llama-sampling │
  │  (encode/     │  │  (temp, top-k, │
  │   decode)     │  │   top-p, rng)  │
  └───────────────┘  └────────────────┘`}
    </div>

    <h2>Workspace Crates</h2>
    <div class="card-grid">
      <a href={`${BASE}/crates/llama-engine`} class="card">
        <div class="card-title">llama-engine</div>
        <div class="card-desc">
          Narrow-waist engine trait and core types. The single stable interface
          that all consumers depend on.
        </div>
        <span class="card-tag">core</span>
      </a>
      <a href={`${BASE}/crates/llama-tokenizer`} class="card">
        <div class="card-title">llama-tokenizer</div>
        <div class="card-desc">
          Deterministic text-to-token conversion with streaming UTF-8 decoding
          and pluggable backends.
        </div>
        <span class="card-tag">tokenization</span>
      </a>
      <a href={`${BASE}/crates/llama-models`} class="card">
        <div class="card-title">llama-models</div>
        <div class="card-desc">
          Model architectures: RMSNorm, RoPE, Attention, MLP (SwiGLU). Safetensors
          weight loading.
        </div>
        <span class="card-tag">models</span>
      </a>
      <a href={`${BASE}/crates/llama-sampling`} class="card">
        <div class="card-title">llama-sampling</div>
        <div class="card-desc">
          Sampling strategies: greedy, temperature, top-k/p, repetition penalty
          with deterministic seeded RNG.
        </div>
        <span class="card-tag">sampling</span>
      </a>
      <a href={`${BASE}/crates/llama-kv`} class="card">
        <div class="card-title">llama-kv</div>
        <div class="card-desc">
          KV cache management: prefill, decode, memory layouts (BySequence, ByHead,
          Transposed). Future paging/eviction.
        </div>
        <span class="card-tag">memory</span>
      </a>
      <a href={`${BASE}/crates/llama-runtime`} class="card">
        <div class="card-title">llama-runtime</div>
        <div class="card-desc">
          Backend selection (CPU/Metal), kernel availability matrix, telemetry hooks
          for TTFT and tokens/sec.
        </div>
        <span class="card-tag">runtime</span>
      </a>
    </div>

    <h2>Key Principles</h2>
    <table>
      <thead>
        <tr>
          <th>Principle</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Narrow waist</strong></td>
          <td>
            <code>LlamaEngine</code> is the single stable interface. Swap
            backends without changing app code.
          </td>
        </tr>
        <tr>
          <td><strong>Quarantined unsafe</strong></td>
          <td>
            All unsafe code lives in small, typed, safe API wrappers.
          </td>
        </tr>
        <tr>
          <td><strong>Resilience by default</strong></td>
          <td>
            Cancellation, timeouts, backpressure, and bounded memory from day one.
          </td>
        </tr>
        <tr>
          <td><strong>Zero-copy IO</strong></td>
          <td>
            Avoid unnecessary copies. Allow mmap for weight loading, expose
            tuning knobs.
          </td>
        </tr>
        <tr>
          <td><strong>Deterministic testing</strong></td>
          <td>
            Seeded RNG, golden tests, KV equivalence verification, and backend
            parity gates.
          </td>
        </tr>
      </tbody>
    </table>
  </Layout>
);
