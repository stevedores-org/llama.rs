import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function LlamaRuntime() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">llama-runtime</h1>
      <p className="text-lg text-zinc-400 mb-6">
        Backend selection, kernel availability probing, and telemetry hooks for performance measurement.
      </p>
      <CodeBlock>{`cargo add llama-runtime                  # default: cpu
cargo add llama-runtime --features metal # enable Metal`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Backend Selection</h2>
      <CodeBlock>{`use llama_runtime::backend::{Backend, BackendSelector};

// Auto-detect best available backend
let selector = BackendSelector::auto()?;
println!("Using: {:?}", selector.backend());

// Or explicitly choose
let selector = BackendSelector::with_backend(Backend::Cpu)?;`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-4 pb-2 border-b border-zinc-800/60">Feature Flags</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-[13px]">
          <thead><tr className="border-b border-zinc-800 bg-zinc-900/50"><th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Feature</th><th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Default</th><th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Description</th></tr></thead>
          <tbody className="text-zinc-400">
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">cpu</td><td className="px-5 py-3">Yes</td><td className="px-5 py-3">CPU backend (always available)</td></tr>
            <tr><td className="px-5 py-3 font-mono text-zinc-300">metal</td><td className="px-5 py-3">No</td><td className="px-5 py-3">Metal GPU backend (macOS only)</td></tr>
          </tbody>
        </table>
      </div>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Kernel Matrix</h2>
      <CodeBlock>{`use llama_runtime::backend::{KernelMatrix, KernelOp, Backend};

let matrix = KernelMatrix::probe();

if matrix.supports(Backend::Metal, KernelOp::Attention) {
    println!("Metal attention available");
}`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Telemetry</h2>
      <CodeBlock>{`use llama_runtime::telemetry::*;

let mut timer = InferenceTimer::new(
    Backend::Cpu,
    prompt_tokens.len(),
    Box::new(LogTelemetry),
);

timer.mark_prefill_complete();
timer.mark_token(); // call per generated token

let metrics = timer.finish();
println!("TTFT: {:.1}ms, {:.1} tok/s",
    metrics.ttft_ms, metrics.tokens_per_sec);`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-4 pb-2 border-b border-zinc-800/60">InferenceMetrics</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-[13px]">
          <thead><tr className="border-b border-zinc-800 bg-zinc-900/50"><th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Field</th><th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Type</th><th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Description</th></tr></thead>
          <tbody className="text-zinc-400">
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">backend</td><td className="px-5 py-3 font-mono">Backend</td><td className="px-5 py-3">Which backend was used</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">ttft_ms</td><td className="px-5 py-3 font-mono">f64</td><td className="px-5 py-3">Time to first token</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">tokens_per_sec</td><td className="px-5 py-3 font-mono">f64</td><td className="px-5 py-3">Generation throughput</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">prompt_tokens</td><td className="px-5 py-3 font-mono">usize</td><td className="px-5 py-3">Number of prompt tokens</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">generated_tokens</td><td className="px-5 py-3 font-mono">usize</td><td className="px-5 py-3">Number of generated tokens</td></tr>
            <tr><td className="px-5 py-3 font-mono text-zinc-300">total_ms</td><td className="px-5 py-3 font-mono">f64</td><td className="px-5 py-3">Total wall-clock time</td></tr>
          </tbody>
        </table>
      </div>

      <Callout icon="🔗">
        <a href="https://crates.io/crates/llama-runtime" className="text-orange-400 hover:text-orange-300">crates.io</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-runtime" className="text-orange-400 hover:text-orange-300">source</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/10" className="text-orange-400 hover:text-orange-300">LLAMA-007</a>
      </Callout>
    </Layout>
  );
}
