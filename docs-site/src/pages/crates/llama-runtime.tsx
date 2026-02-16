import type { FC } from "hono/jsx";
import { Layout } from "../../components/Layout";

export const LlamaRuntimePage: FC = () => (
  <Layout title="llama-runtime" activePath="/crates/llama-runtime">
    <h1>llama-runtime</h1>
    <p class="lead">
      Backend selection, kernel availability probing, and telemetry hooks for
      performance measurement. The execution layer for llama.rs.
    </p>
    <pre><code>{`cargo add llama-runtime                  # default: cpu
cargo add llama-runtime --features metal # enable Metal`}</code></pre>

    <h2>Backend Selection</h2>
    <p>
      Backends are controlled via feature flags and auto-detected at runtime:
    </p>
    <pre>
      <code>
        <span class="kw">use</span>{" llama_runtime::backend::{Backend, BackendSelector};\n\n"}
        <span class="cm">{"// Auto-detect best available backend\n"}</span>
        <span class="kw">let</span>{" selector = BackendSelector::auto()?;\n"}
        {"println!("}<span class="st">{'"Using: {:?}"'}</span>{", selector.backend());\n\n"}
        <span class="cm">{"// Or explicitly choose\n"}</span>
        <span class="kw">let</span>{" selector = BackendSelector::with_backend(Backend::Cpu)?;\n"}
      </code>
    </pre>

    <h2>Feature Flags</h2>
    <table>
      <thead>
        <tr><th>Feature</th><th>Default</th><th>Description</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><code>cpu</code></td>
          <td>Yes</td>
          <td>CPU backend (always available)</td>
        </tr>
        <tr>
          <td><code>metal</code></td>
          <td>No</td>
          <td>Metal GPU backend (macOS only)</td>
        </tr>
      </tbody>
    </table>

    <h2>Kernel Matrix</h2>
    <p>
      The runtime probes which operations each backend supports at startup:
    </p>
    <pre>
      <code>
        <span class="kw">use</span>{" llama_runtime::backend::{KernelMatrix, KernelOp, Backend};\n\n"}
        <span class="kw">let</span>{" matrix = KernelMatrix::probe();\n\n"}
        <span class="cm">{"// Check if a specific op is supported\n"}</span>
        <span class="kw">if</span>{" matrix.supports(Backend::Metal, KernelOp::Attention) {\n"}
        {"    println!("}<span class="st">{'"Metal attention available"'}</span>{");\n"}
        {"}\n"}
      </code>
    </pre>
    <p>Operations probed:</p>
    <table>
      <thead>
        <tr><th>KernelOp</th><th>CPU</th><th>Metal</th></tr>
      </thead>
      <tbody>
        <tr><td><code>RmsNorm</code></td><td>Always</td><td>macOS only</td></tr>
        <tr><td><code>Rope</code></td><td>Always</td><td>macOS only</td></tr>
        <tr><td><code>Attention</code></td><td>Always</td><td>macOS only</td></tr>
        <tr><td><code>MatMul</code></td><td>Always</td><td>macOS only</td></tr>
        <tr><td><code>Softmax</code></td><td>Always</td><td>macOS only</td></tr>
        <tr><td><code>SwiGLU</code></td><td>Always</td><td>macOS only</td></tr>
      </tbody>
    </table>

    <h2>Telemetry</h2>
    <p>
      Hook into inference lifecycle for performance measurement:
    </p>
    <pre>
      <code>
        <span class="kw">use</span>{" llama_runtime::telemetry::*;\n\n"}
        <span class="cm">{"// Built-in logging hook\n"}</span>
        <span class="kw">let</span>{" hook = LogTelemetry;\n\n"}
        <span class="cm">{"// Start timing an inference run\n"}</span>
        <span class="kw">let mut</span>{" timer = InferenceTimer::new(\n"}
        {"    Backend::Cpu,\n"}
        {"    prompt_tokens.len(),\n"}
        {"    Box::new(hook),\n"}
        {");\n\n"}
        <span class="cm">{"// Mark events\n"}</span>
        {"timer.mark_prefill_complete();\n"}
        {"timer.mark_token(); "}<span class="cm">{"// call per generated token\n"}</span>
        {"\n"}
        <span class="cm">{"// Finalize and get metrics\n"}</span>
        <span class="kw">let</span>{" metrics = timer.finish();\n"}
        {"println!("}<span class="st">{'"TTFT: {:.1}ms, {:.1} tok/s"'}</span>{",\n"}
        {"    metrics.ttft_ms, metrics.tokens_per_sec);\n"}
      </code>
    </pre>

    <h3>InferenceMetrics</h3>
    <table>
      <thead>
        <tr><th>Field</th><th>Type</th><th>Description</th></tr>
      </thead>
      <tbody>
        <tr><td><code>backend</code></td><td><code>Backend</code></td><td>Which backend was used</td></tr>
        <tr><td><code>ttft_ms</code></td><td><code>f64</code></td><td>Time to first token</td></tr>
        <tr><td><code>tokens_per_sec</code></td><td><code>f64</code></td><td>Generation throughput</td></tr>
        <tr><td><code>prompt_tokens</code></td><td><code>usize</code></td><td>Number of prompt tokens</td></tr>
        <tr><td><code>generated_tokens</code></td><td><code>usize</code></td><td>Number of generated tokens</td></tr>
        <tr><td><code>total_ms</code></td><td><code>f64</code></td><td>Total wall-clock time</td></tr>
      </tbody>
    </table>

    <h2>Verification</h2>
    <p>
      The crate includes a <code>RuntimeVerifier</code> for LLAMA-006 true
      tests: verifying that <code>full_forward(prompt)</code> logits match{" "}
      <code>prefill(prompt[:-1]) + decode(last_token)</code> logits.
    </p>

    <div class="callout">
      <span class="callout-icon">&#x1F517;</span>
      <div>
        <a href="https://crates.io/crates/llama-runtime">crates.io</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-runtime">source</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/10">LLAMA-007</a>
      </div>
    </div>
  </Layout>
);
