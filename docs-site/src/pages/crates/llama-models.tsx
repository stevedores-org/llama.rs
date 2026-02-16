import type { FC } from "hono/jsx";
import { Layout } from "../../components/Layout";

export const LlamaModelsPage: FC = () => (
  <Layout title="llama-models" activePath="/crates/llama-models">
    <h1>llama-models</h1>
    <p class="lead">
      Foundational model building blocks for Llama, Qwen, and Mistral
      architectures. Includes weight loading from safetensors format.
    </p>
    <pre><code>cargo add llama-models</code></pre>

    <h2>Building Blocks</h2>
    <div class="card-grid">
      <div class="card">
        <div class="card-title">rms_norm</div>
        <div class="card-desc">
          Root Mean Square Layer Normalization. Takes input, weight vector, and
          epsilon. Returns normalized output.
        </div>
        <span class="card-tag">normalization</span>
      </div>
      <div class="card">
        <div class="card-title">apply_rope</div>
        <div class="card-desc">
          Rotary Positional Embeddings (RoPE). In-place application to Q and K
          vectors for a given position.
        </div>
        <span class="card-tag">positional</span>
      </div>
      <div class="card">
        <div class="card-title">attention_decode</div>
        <div class="card-desc">
          Scaled dot-product attention for single-step decode. Q against all
          cached K, V with multi-head support.
        </div>
        <span class="card-tag">attention</span>
      </div>
      <div class="card">
        <div class="card-title">mlp_swiglu</div>
        <div class="card-desc">
          SwiGLU MLP block: gate, up, and down projections with the SiLU
          activation gating mechanism.
        </div>
        <span class="card-tag">feedforward</span>
      </div>
    </div>

    <h2>API Reference</h2>
    <pre>
      <code>
        <span class="cm">{"// RMSNorm\n"}</span>
        <span class="kw">pub fn</span>{" "}<span class="fn">rms_norm</span>{"(\n"}
        {"    x: &[f32], weight: &[f32], eps: f32\n"}
        {") -> Result<Vec<f32>, ModelError>\n\n"}
        <span class="cm">{"// RoPE (in-place)\n"}</span>
        <span class="kw">pub fn</span>{" "}<span class="fn">apply_rope</span>{"(\n"}
        {"    q: &mut [f32], k: &mut [f32],\n"}
        {"    position: usize, n_heads: usize, head_dim: usize, base: f32\n"}
        {") -> Result<(), ModelError>\n\n"}
        <span class="cm">{"// Decode attention\n"}</span>
        <span class="kw">pub fn</span>{" "}<span class="fn">attention_decode</span>{"(\n"}
        {"    q: &[f32], keys: &[f32], values: &[f32],\n"}
        {"    seq_len: usize, n_heads: usize, head_dim: usize\n"}
        {") -> Result<Vec<f32>, ModelError>\n\n"}
        <span class="cm">{"// SwiGLU MLP\n"}</span>
        <span class="kw">pub fn</span>{" "}<span class="fn">mlp_swiglu</span>{"(\n"}
        {"    x: &[f32], w_gate: &[f32], w_up: &[f32], w_down: &[f32],\n"}
        {"    d_model: usize, d_ff: usize\n"}
        {") -> Result<Vec<f32>, ModelError>\n"}
      </code>
    </pre>

    <h2>Weight Loading</h2>
    <p>
      Load model weights from the safetensors format:
    </p>
    <pre>
      <code>
        <span class="kw">let</span>{" weights = ModelWeights::load_safetensors("}<span class="st">{'"model.safetensors"'}</span>{")?\n\n"}
        <span class="cm">{"// Access named tensors\n"}</span>
        <span class="kw">let</span>{" norm = weights.get("}<span class="st">{'"model.norm.weight"'}</span>{")?\n"}
        {"println!("}<span class="st">{'"shape: {:?}, len: {}"'}</span>{", norm.shape, norm.data.len());\n"}
      </code>
    </pre>
    <p>
      Currently supports F32 tensors only. Quantized formats (Q4, Q8) are
      planned for later milestones with separate kernel paths.
    </p>

    <div class="callout">
      <span class="callout-icon">&#x1F517;</span>
      <div>
        <a href="https://crates.io/crates/llama-models">crates.io</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-models">source</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/8">LLAMA-005</a>
      </div>
    </div>
  </Layout>
);
