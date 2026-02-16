import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function Architecture() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">Architecture</h1>
      <p className="text-lg text-zinc-400 mb-10">
        A modular, resilient, and accelerated inference runtime in Rust.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-4 pb-2 border-b border-zinc-800/60">Design Principles</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900/50">
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Principle</th>
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">How</th>
            </tr>
          </thead>
          <tbody className="text-zinc-400">
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">Narrow waist</td><td className="px-5 py-3">LlamaEngine trait is the single stable interface</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">Quarantined unsafe</td><td className="px-5 py-3">Unsafe code wrapped in typed, safe APIs</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">API stability</td><td className="px-5 py-3">Public interfaces are narrow, versioned, and testable</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">Resilience by default</td><td className="px-5 py-3">Cancellation, timeouts, backpressure, bounded memory</td></tr>
            <tr><td className="px-5 py-3 text-zinc-200 font-medium">Performance</td><td className="px-5 py-3">Zero-copy IO (mmap), avoid unnecessary copies, tuning knobs</td></tr>
          </tbody>
        </table>
      </div>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">The Narrow Waist: LlamaEngine</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        Everything plugs into the LlamaEngine trait. Methods take{" "}
        <code className="font-mono text-orange-300/90 text-[13px]">&self</code> to allow
        shared access across sessions and concurrent inference.
      </p>
      <CodeBlock title="llama-engine/src/lib.rs">{`pub trait LlamaEngine: Send + Sync {
    fn load_model(&self, spec: &ModelSpec) -> Result<ModelHandle>;
    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;
    fn detokenize(&self, tokens: &[i32]) -> Result<String>;
    fn prefill(&self, session: &mut Session, tokens: &[i32]) -> Result<PrefillResult>;
    fn decode(&self, session: &mut Session) -> Result<TokenStream>;
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Crate Layout</h2>
      <pre className="bg-zinc-900 border border-zinc-800 rounded-xl px-5 py-4 overflow-x-auto text-[12px] leading-relaxed font-mono text-zinc-400 my-4">
{`llama.rs/
  crates/
    `}<span className="text-orange-400 font-semibold">llama-engine/</span>{`        # narrow-waist trait + streaming API
    llama-models/        # model architectures (Llama/Qwen/Mistral)
    `}<span className="text-orange-400 font-semibold">llama-runtime/</span>{`       # execution: oxidizedMLX, backend selection
    llama-tokenizer/     # tokenizers + chat templates
    llama-sampling/      # samplers + penalties + stop conditions
    llama-kv/            # KV cache layouts + paging/eviction`}
      </pre>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Future Crates</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900/50">
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Crate</th>
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Purpose</th>
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Story</th>
            </tr>
          </thead>
          <tbody className="text-zinc-400">
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">llama-server</td><td className="px-5 py-3">OpenAI-compatible HTTP API</td><td className="px-5 py-3">LLAMA-009</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">llama-cli</td><td className="px-5 py-3">CLI runner for debug + bench</td><td className="px-5 py-3">Milestone A</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">llama-rag</td><td className="px-5 py-3">oxidizedRAG adapter</td><td className="px-5 py-3">LLAMA-011</td></tr>
            <tr><td className="px-5 py-3 font-mono text-zinc-300">llama-agents</td><td className="px-5 py-3">oxidizedgraph agent nodes</td><td className="px-5 py-3">LLAMA-012</td></tr>
          </tbody>
        </table>
      </div>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Acceleration Strategy</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        The runtime exposes capability discovery rather than hard-coding platform rules.
        Feature gates control backend compilation; a kernel matrix validates op support at startup.
      </p>
      <CodeBlock>{`// Feature gates control backend compilation
#[cfg(feature = "cpu")]  Cpu,
#[cfg(feature = "metal")] Metal,

// Kernel availability matrix probes op support
let matrix = KernelMatrix::probe();
let selector = BackendSelector::auto()?;`}</CodeBlock>

      <Callout icon="🛡️">
        <strong className="text-zinc-100">Backend parity gate:</strong> Metal cannot
        become the default backend unless parity tests pass against the CPU
        reference implementation.
      </Callout>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Session Lifecycle</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900/50">
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Property</th>
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Design</th>
            </tr>
          </thead>
          <tbody className="text-zinc-400">
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">Cancellation</td><td className="px-5 py-3">Cooperative. Dropping a TokenStream terminates graph execution.</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">Memory</td><td className="px-5 py-3">Bounded. Context size and KV cache behavior are explicit and configurable.</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">Lifecycle</td><td className="px-5 py-3">Session ID maps to KV cache lifecycle (freed or archived on completion).</td></tr>
            <tr><td className="px-5 py-3 text-zinc-200 font-medium">Cloning</td><td className="px-5 py-3">Intentionally not Clone — duplicating KV cache state is not cheap.</td></tr>
          </tbody>
        </table>
      </div>
    </Layout>
  );
}
