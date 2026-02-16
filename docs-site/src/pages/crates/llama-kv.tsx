import type { FC } from "hono/jsx";
import { Layout } from "../../components/Layout";

export const LlamaKvPage: FC = () => (
  <Layout title="llama-kv" activePath="/crates/llama-kv">
    <h1>llama-kv</h1>
    <p class="lead">
      First-class KV cache management for transformer inference. Supports
      prefill, decode, multiple memory layouts, and multi-layer sessions.
    </p>
    <pre><code>cargo add llama-kv</code></pre>

    <h2>Cache Architecture</h2>
    <p>
      The KV cache stores key and value projections for all processed tokens,
      enabling efficient autoregressive generation without recomputing
      attention over the full sequence.
    </p>
    <div class="arch-diagram">
{`Prefill (N tokens)           Decode (1 token)
┌─────────────────┐         ┌──────────┐
│ K: [N, H, D]    │         │ K: +1    │
│ V: [N, H, D]    │  ───►   │ V: +1    │
└─────────────────┘         └──────────┘
     seq_len = N             seq_len = N+1`}
    </div>

    <h2>Memory Layouts</h2>
    <table>
      <thead>
        <tr><th>Layout</th><th>Memory Order</th><th>Use Case</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><code>BySequence</code></td>
          <td><code>[seq_len][heads][head_dim]</code></td>
          <td>Good for positional access during decode</td>
        </tr>
        <tr>
          <td><code>ByHead</code></td>
          <td><code>[heads][seq_len][head_dim]</code></td>
          <td>Good for per-head operations, Metal alignment</td>
        </tr>
        <tr>
          <td><code>Transposed</code></td>
          <td><code>[heads][head_dim][seq_len]</code></td>
          <td>Optimizes attention Q*K^T computation</td>
        </tr>
      </tbody>
    </table>

    <h2>Layer Cache</h2>
    <pre>
      <code>
        <span class="kw">use</span>{" llama_kv::{LayerKVCache, KVLayout};\n\n"}
        <span class="kw">let mut</span>{" cache = LayerKVCache::new(\n"}
        {"    "}<span class="nr">{"64"}</span>{",   "}<span class="cm">{"// max_seq_len\n"}</span>
        {"    "}<span class="nr">{"32"}</span>{",   "}<span class="cm">{"// n_heads\n"}</span>
        {"    "}<span class="nr">{"128"}</span>{",  "}<span class="cm">{"// head_dim\n"}</span>
        {"    KVLayout::BySequence,\n"}
        {");\n\n"}
        <span class="cm">{"// Append token K/V vectors\n"}</span>
        {"cache.append_token(&k_vec, &v_vec)?;\n"}
        <span class="mc">assert_eq!</span>{"(cache.seq_len, "}<span class="nr">{"1"}</span>{");\n"}
      </code>
    </pre>

    <h2>Multi-Layer Sessions</h2>
    <p>
      <code>SessionKVCache</code> manages KV caches across all transformer
      layers with atomic append operations:
    </p>
    <pre>
      <code>
        <span class="kw">use</span>{" llama_kv::SessionKVCache;\n\n"}
        <span class="kw">let mut</span>{" session = SessionKVCache::new(\n"}
        {"    "}<span class="nr">{"32"}</span>{",  "}<span class="cm">{"// n_layers\n"}</span>
        {"    "}<span class="nr">{"64"}</span>{",  "}<span class="cm">{"// max_seq_len\n"}</span>
        {"    "}<span class="nr">{"32"}</span>{",  "}<span class="cm">{"// n_heads\n"}</span>
        {"    "}<span class="nr">{"128"}</span>{", "}<span class="cm">{"// head_dim\n"}</span>
        {"    KVLayout::BySequence,\n"}
        {");\n\n"}
        <span class="cm">{"// Atomic append across all layers\n"}</span>
        {"session.append_token(&k_per_layer, &v_per_layer)?;\n"}
        <span class="mc">assert_eq!</span>{"(session.seq_len(), "}<span class="nr">{"1"}</span>{");\n"}
      </code>
    </pre>

    <h2>Shape &amp; Capacity</h2>
    <pre>
      <code>
        <span class="kw">use</span>{" llama_kv::KVShape;\n\n"}
        <span class="kw">let</span>{" shape = KVShape::new("}<span class="nr">{"1024"}</span>{", "}<span class="nr">{"32"}</span>{", "}<span class="nr">{"128"}</span>{");\n"}
        {"println!("}<span class="st">{'"Total elements: {}"'}</span>{", shape.total_elements());\n"}
        {"println!("}<span class="st">{'"Bytes (f16): {}"'}</span>{", shape.capacity_bytes("}<span class="nr">{"2"}</span>{"));\n"}
      </code>
    </pre>

    <div class="callout">
      <span class="callout-icon">&#x1F517;</span>
      <div>
        <a href="https://crates.io/crates/llama-kv">crates.io</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-kv">source</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/7">LLAMA-004</a>
      </div>
    </div>
  </Layout>
);
