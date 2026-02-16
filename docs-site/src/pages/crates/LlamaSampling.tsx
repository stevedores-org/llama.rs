import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function LlamaSampling() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">llama-sampling</h1>
      <p className="text-lg text-zinc-400 mb-6">
        Sampling and decoding strategies with deterministic seeded RNG for reproducible generation.
      </p>
      <CodeBlock>cargo add llama-sampling</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-4 pb-2 border-b border-zinc-800/60">Strategies</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-[13px]">
          <thead><tr className="border-b border-zinc-800 bg-zinc-900/50"><th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Strategy</th><th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Description</th></tr></thead>
          <tbody className="text-zinc-400">
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">Greedy</td><td className="px-5 py-3">Argmax. Always picks the highest-probability token.</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">Temperature</td><td className="px-5 py-3">Scale logits by 1/T before softmax.</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">Top-k</td><td className="px-5 py-3">Keep only top k tokens by probability.</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">Top-p (nucleus)</td><td className="px-5 py-3">Smallest set with cumulative probability &ge; p.</td></tr>
            <tr><td className="px-5 py-3 text-zinc-200 font-medium">Repetition Penalty</td><td className="px-5 py-3">Penalize tokens in history. Handles negative logits correctly.</td></tr>
          </tbody>
        </table>
      </div>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Usage</h2>
      <CodeBlock>{`use llama_sampling::{Sampler, SamplingConfig, SamplingStrategy};

let mut sampler = Sampler::new(SamplingConfig {
    strategy: SamplingStrategy::Stochastic,
    temperature: 0.8,
    top_k: 40,
    top_p: 0.95,
    repetition_penalty: 1.1,
    seed: 42,
    ..SamplingConfig::default()
})?;

// Sample with generation history for repetition penalty
let token = sampler.sample(&logits, &history)?;`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-4 pb-2 border-b border-zinc-800/60">Configuration</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-[13px]">
          <thead><tr className="border-b border-zinc-800 bg-zinc-900/50"><th className="text-left px-4 py-2.5 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Field</th><th className="text-left px-4 py-2.5 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Type</th><th className="text-left px-4 py-2.5 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Default</th><th className="text-left px-4 py-2.5 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Notes</th></tr></thead>
          <tbody className="text-zinc-400">
            <tr className="border-b border-zinc-800/60"><td className="px-4 py-2.5 font-mono text-zinc-300">strategy</td><td className="px-4 py-2.5 font-mono">SamplingStrategy</td><td className="px-4 py-2.5">Greedy</td><td className="px-4 py-2.5">Greedy or Stochastic</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-4 py-2.5 font-mono text-zinc-300">temperature</td><td className="px-4 py-2.5 font-mono">f32</td><td className="px-4 py-2.5">1.0</td><td className="px-4 py-2.5">Must be &gt; 0</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-4 py-2.5 font-mono text-zinc-300">top_k</td><td className="px-4 py-2.5 font-mono">usize</td><td className="px-4 py-2.5">0</td><td className="px-4 py-2.5">0 = disabled</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-4 py-2.5 font-mono text-zinc-300">top_p</td><td className="px-4 py-2.5 font-mono">f32</td><td className="px-4 py-2.5">1.0</td><td className="px-4 py-2.5">1.0 = disabled</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-4 py-2.5 font-mono text-zinc-300">repetition_penalty</td><td className="px-4 py-2.5 font-mono">f32</td><td className="px-4 py-2.5">1.0</td><td className="px-4 py-2.5">1.0 = disabled</td></tr>
            <tr><td className="px-4 py-2.5 font-mono text-zinc-300">seed</td><td className="px-4 py-2.5 font-mono">u64</td><td className="px-4 py-2.5">0</td><td className="px-4 py-2.5">0 auto-maps to 1</td></tr>
          </tbody>
        </table>
      </div>

      <Callout icon="🔗">
        <a href="https://crates.io/crates/llama-sampling" className="text-orange-400 hover:text-orange-300">crates.io</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-sampling" className="text-orange-400 hover:text-orange-300">source</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/6" className="text-orange-400 hover:text-orange-300">LLAMA-003</a>
      </Callout>
    </Layout>
  );
}
