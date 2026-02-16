import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import { CardGrid } from "@/components/Card";
import Callout from "@/components/Callout";

export default function LlamaModels() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">llama-models</h1>
      <p className="text-lg text-zinc-400 mb-6">
        Foundational model building blocks for Llama, Qwen, and Mistral architectures with safetensors weight loading.
      </p>
      <CodeBlock>cargo add llama-models</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-4 pb-2 border-b border-zinc-800/60">Building Blocks</h2>
      <CardGrid>
        {[
          { title: "rms_norm", description: "Root Mean Square Layer Normalization", tag: "normalization" },
          { title: "apply_rope", description: "Rotary Positional Embeddings (in-place on Q and K)", tag: "positional" },
          { title: "attention_decode", description: "Scaled dot-product attention for single-step decode", tag: "attention" },
          { title: "mlp_swiglu", description: "SwiGLU MLP: gate, up, down projections with SiLU gating", tag: "feedforward" },
        ].map((b) => (
          <div key={b.title} className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
            <div className="font-mono text-sm font-semibold text-orange-400 mb-2">{b.title}</div>
            <div className="text-[13px] text-zinc-400 leading-relaxed">{b.description}</div>
            <span className="inline-block mt-3 text-[10px] font-semibold text-zinc-500 bg-zinc-950 border border-zinc-800 rounded px-2 py-0.5 uppercase tracking-wider">{b.tag}</span>
          </div>
        ))}
      </CardGrid>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">API</h2>
      <CodeBlock>{`pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, ModelError>

pub fn apply_rope(
    q: &mut [f32], k: &mut [f32],
    position: usize, n_heads: usize, head_dim: usize, base: f32
) -> Result<(), ModelError>

pub fn attention_decode(
    q: &[f32], keys: &[f32], values: &[f32],
    seq_len: usize, n_heads: usize, head_dim: usize
) -> Result<Vec<f32>, ModelError>

pub fn mlp_swiglu(
    x: &[f32], w_gate: &[f32], w_up: &[f32], w_down: &[f32],
    d_model: usize, d_ff: usize
) -> Result<Vec<f32>, ModelError>`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Weight Loading</h2>
      <CodeBlock>{`let weights = ModelWeights::load_safetensors("model.safetensors")?;
let norm = weights.get("model.norm.weight")?;
println!("shape: {:?}, len: {}", norm.shape, norm.data.len());`}</CodeBlock>
      <p className="text-zinc-400 text-[15px] mb-4">
        Currently supports F32 tensors only. Quantized formats (Q4, Q8) planned for later milestones.
      </p>

      <Callout icon="🔗">
        <a href="https://crates.io/crates/llama-models" className="text-orange-400 hover:text-orange-300">crates.io</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-models" className="text-orange-400 hover:text-orange-300">source</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/8" className="text-orange-400 hover:text-orange-300">LLAMA-005</a>
      </Callout>
    </Layout>
  );
}
