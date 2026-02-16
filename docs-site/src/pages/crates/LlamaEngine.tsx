import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function LlamaEngine() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">llama-engine</h1>
      <p className="text-lg text-zinc-400 mb-6">
        The "narrow waist" of the llama.rs stack. Core trait and types that all other crates depend on.
      </p>
      <CodeBlock>cargo add llama-engine</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Core Trait</h2>
      <CodeBlock title="LlamaEngine">{`pub trait LlamaEngine: Send + Sync {
    fn load_model(&self, spec: &ModelSpec) -> Result<ModelHandle>;
    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;
    fn detokenize(&self, tokens: &[i32]) -> Result<String>;
    fn prefill(&self, session: &mut Session, tokens: &[i32]) -> Result<PrefillResult>;
    fn decode(&self, session: &mut Session) -> Result<TokenStream>;
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}`}</CodeBlock>
      <p className="text-zinc-400 text-[15px] mb-4">
        Methods take <code className="font-mono text-orange-300/90 text-[13px]">&self</code>{" "}
        to allow shared access via Arc. Backends use interior mutability for synchronization.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-4 pb-2 border-b border-zinc-800/60">Key Types</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900/50">
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Type</th>
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Description</th>
            </tr>
          </thead>
          <tbody className="text-zinc-400">
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">TokenId</td><td className="px-5 py-3">Alias for i32 (FFI compatible). Logically non-negative.</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">ModelSpec</td><td className="px-5 py-3">Configuration for loading a model (path, context size).</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">ModelHandle</td><td className="px-5 py-3">Opaque handle to a loaded model. No public fields.</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">Session</td><td className="px-5 py-3">Active inference session with KV cache state. Not Clone.</td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 font-mono text-zinc-300">PrefillResult</td><td className="px-5 py-3">Result of prefill phase. Marked #[must_use].</td></tr>
            <tr><td className="px-5 py-3 font-mono text-zinc-300">LlamaError</td><td className="px-5 py-3">Typed error enum: ModelLoad, Tokenization, Inference.</td></tr>
          </tbody>
        </table>
      </div>

      <Callout icon="🔗">
        <a href="https://crates.io/crates/llama-engine" className="text-orange-400 hover:text-orange-300">crates.io</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-engine" className="text-orange-400 hover:text-orange-300">source</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/4" className="text-orange-400 hover:text-orange-300">LLAMA-001</a>
      </Callout>
    </Layout>
  );
}
