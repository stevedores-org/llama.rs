import { Link } from "react-router-dom";
import Layout from "@/components/Layout";
import { Card, CardGrid } from "@/components/Card";
import Callout from "@/components/Callout";

export default function Home() {
  return (
    <Layout>
      {/* Hero */}
      <div className="pb-10 mb-10 border-b border-zinc-800/60">
        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-gradient-to-br from-zinc-100 to-zinc-400 bg-clip-text text-transparent leading-tight">
          llama.rs
        </h1>
        <p className="text-lg text-zinc-400 mt-3 leading-relaxed max-w-xl">
          A modular Rust inference runtime for Llama-family models. Built on{" "}
          <a href="https://github.com/stevedores-org/oxidizedMLX" className="text-orange-400 hover:text-orange-300">
            oxidizedMLX
          </a>{" "}
          for Metal/CPU acceleration.
        </p>
        <div className="flex gap-3 mt-6">
          <Link
            to="/getting-started"
            className="bg-orange-500 hover:bg-orange-600 text-black font-semibold px-5 py-2.5 rounded-lg transition text-sm"
          >
            Get Started
          </Link>
          <a
            href="https://github.com/stevedores-org/llama.rs"
            className="border border-zinc-700 hover:border-zinc-500 px-5 py-2.5 rounded-lg transition text-sm text-zinc-300"
          >
            GitHub
          </a>
          <a
            href="https://crates.io/crates/llama-engine"
            className="border border-zinc-700 hover:border-zinc-500 px-5 py-2.5 rounded-lg transition text-sm text-zinc-300"
          >
            crates.io
          </a>
        </div>
      </div>

      {/* Narrow waist */}
      <h2 className="text-2xl font-bold tracking-tight mb-3">Design Philosophy</h2>
      <p className="text-zinc-400 text-[15px] leading-relaxed mb-4">
        llama.rs uses a <strong className="text-zinc-200">"narrow waist"</strong>{" "}
        design: the <code className="font-mono text-orange-300/90 text-[13px]">llama-engine</code>{" "}
        crate defines the core <code className="font-mono text-orange-300/90 text-[13px]">LlamaEngine</code>{" "}
        trait that all other crates depend on. Swap CPU/Metal/FFI backends
        without changing application code.
      </p>

      <Callout icon="⚡">
        <strong className="text-zinc-100">Performance is a feature.</strong>{" "}
        Zero-copy IO via mmap, bounded memory via explicit KV cache policies,
        and cooperative cancellation from day one.
      </Callout>

      {/* Architecture */}
      <pre className="bg-zinc-900 border border-zinc-800 rounded-xl px-6 py-5 overflow-x-auto text-[12px] leading-relaxed font-mono text-zinc-400 my-8">
{`                    ┌──────────────────────┐
                    │    Application       │
                    │  (llama-cli, server)  │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │    `}<span className="text-orange-400 font-semibold">llama-engine</span>{`       │
                    │   (narrow waist)     │
                    └──┬───────┬───────┬───┘
                       │       │       │
          ┌────────────┘       │       └────────────┐
          ▼                    ▼                    ▼
  ┌───────────────┐  ┌────────────────┐  ┌───────────────┐
  │ llama-models  │  │ llama-runtime  │  │  llama-kv     │
  └───────────────┘  └────────────────┘  └───────────────┘
          │                    │
  ┌───────────────┐  ┌────────────────┐
  │llama-tokenizer│  │ llama-sampling │
  └───────────────┘  └────────────────┘`}
      </pre>

      {/* Crate cards */}
      <h2 className="text-2xl font-bold tracking-tight mt-12 mb-3">Workspace Crates</h2>
      <CardGrid>
        <Card to="/crates/llama-engine" title="llama-engine" tag="core"
          description="Narrow-waist engine trait and core types. The single stable interface all consumers depend on." />
        <Card to="/crates/llama-tokenizer" title="llama-tokenizer" tag="tokenization"
          description="Deterministic text-to-token conversion with streaming UTF-8 decoding and pluggable backends." />
        <Card to="/crates/llama-models" title="llama-models" tag="models"
          description="Model architectures: RMSNorm, RoPE, Attention, MLP (SwiGLU). Safetensors weight loading." />
        <Card to="/crates/llama-sampling" title="llama-sampling" tag="sampling"
          description="Sampling strategies: greedy, temperature, top-k/p, repetition penalty with deterministic RNG." />
        <Card to="/crates/llama-kv" title="llama-kv" tag="memory"
          description="KV cache management: prefill, decode, memory layouts (BySequence, ByHead, Transposed)." />
        <Card to="/crates/llama-runtime" title="llama-runtime" tag="runtime"
          description="Backend selection (CPU/Metal), kernel availability matrix, telemetry hooks for TTFT/tok/s." />
      </CardGrid>

      {/* Principles table */}
      <h2 className="text-2xl font-bold tracking-tight mt-12 mb-4">Key Principles</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900/50">
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Principle</th>
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Description</th>
            </tr>
          </thead>
          <tbody className="text-zinc-400">
            <tr className="border-b border-zinc-800/60 hover:bg-zinc-900/30">
              <td className="px-5 py-3 text-zinc-200 font-medium">Narrow waist</td>
              <td className="px-5 py-3">LlamaEngine is the single stable interface. Swap backends without changing app code.</td>
            </tr>
            <tr className="border-b border-zinc-800/60 hover:bg-zinc-900/30">
              <td className="px-5 py-3 text-zinc-200 font-medium">Quarantined unsafe</td>
              <td className="px-5 py-3">All unsafe code lives in small, typed, safe API wrappers.</td>
            </tr>
            <tr className="border-b border-zinc-800/60 hover:bg-zinc-900/30">
              <td className="px-5 py-3 text-zinc-200 font-medium">Resilience by default</td>
              <td className="px-5 py-3">Cancellation, timeouts, backpressure, and bounded memory from day one.</td>
            </tr>
            <tr className="border-b border-zinc-800/60 hover:bg-zinc-900/30">
              <td className="px-5 py-3 text-zinc-200 font-medium">Zero-copy IO</td>
              <td className="px-5 py-3">Avoid unnecessary copies. Allow mmap for weight loading, expose tuning knobs.</td>
            </tr>
            <tr className="hover:bg-zinc-900/30">
              <td className="px-5 py-3 text-zinc-200 font-medium">Deterministic testing</td>
              <td className="px-5 py-3">Seeded RNG, golden tests, KV equivalence verification, and backend parity gates.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Layout>
  );
}
