import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function LlamaKv() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">llama-kv</h1>
      <p className="text-lg text-zinc-400 mb-6">
        First-class KV cache management: prefill, decode, memory layouts, and multi-layer sessions.
      </p>
      <CodeBlock>cargo add llama-kv</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Cache Architecture</h2>
      <pre className="bg-zinc-900 border border-zinc-800 rounded-xl px-5 py-4 overflow-x-auto text-[12px] leading-relaxed font-mono text-zinc-400 my-4">
{`Prefill (N tokens)           Decode (1 token)
┌─────────────────┐         ┌──────────┐
│ K: [N, H, D]    │         │ K: +1    │
│ V: [N, H, D]    │  ───►   │ V: +1    │
└─────────────────┘         └──────────┘
     seq_len = N             seq_len = N+1`}
      </pre>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-4 pb-2 border-b border-zinc-800/60">Memory Layouts</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-[13px]">
          <thead><tr className="border-b border-zinc-800 bg-zinc-900/50"><th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Layout</th><th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Memory Order</th><th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Use Case</th></tr></thead>
          <tbody className="text-zinc-400">
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">BySequence</td><td className="px-5 py-3 font-mono">[seq][heads][dim]</td><td className="px-5 py-3">Positional access during decode</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">ByHead</td><td className="px-5 py-3 font-mono">[heads][seq][dim]</td><td className="px-5 py-3">Per-head operations, Metal alignment</td></tr>
            <tr><td className="px-5 py-3 font-mono text-zinc-300">Transposed</td><td className="px-5 py-3 font-mono">[heads][dim][seq]</td><td className="px-5 py-3">Optimizes Q*K^T attention</td></tr>
          </tbody>
        </table>
      </div>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Layer Cache</h2>
      <CodeBlock>{`use llama_kv::{LayerKVCache, KVLayout};

let mut cache = LayerKVCache::new(
    64,  // max_seq_len
    32,  // n_heads
    128, // head_dim
    KVLayout::BySequence,
);

cache.append_token(&k_vec, &v_vec)?;
assert_eq!(cache.seq_len, 1);`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Multi-Layer Sessions</h2>
      <CodeBlock>{`use llama_kv::SessionKVCache;

let mut session = SessionKVCache::new(
    32,  // n_layers
    64,  // max_seq_len
    32,  // n_heads
    128, // head_dim
    KVLayout::BySequence,
);

// Atomic append across all layers
session.append_token(&k_per_layer, &v_per_layer)?;
assert_eq!(session.seq_len(), 1);`}</CodeBlock>

      <Callout icon="🔗">
        <a href="https://crates.io/crates/llama-kv" className="text-orange-400 hover:text-orange-300">crates.io</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-kv" className="text-orange-400 hover:text-orange-300">source</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/7" className="text-orange-400 hover:text-orange-300">LLAMA-004</a>
      </Callout>
    </Layout>
  );
}
