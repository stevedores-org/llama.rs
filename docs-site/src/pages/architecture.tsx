import type { FC } from "hono/jsx";
import { Layout } from "../components/Layout";

export const ArchitecturePage: FC = () => (
  <Layout title="Architecture" activePath="/architecture">
    <h1>Architecture</h1>
    <p class="lead">
      A modular, resilient, and accelerated inference runtime in Rust.
      Built with oxidizedMLX for tensor/Metal acceleration, oxidizedRAG for
      retrieval, and oxidizedgraph for agent orchestration.
    </p>

    <h2>Design Principles</h2>
    <table>
      <thead>
        <tr>
          <th>Principle</th>
          <th>How</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Narrow waist</strong></td>
          <td>
            <code>LlamaEngine</code> trait is the single stable interface all
            consumers depend on.
          </td>
        </tr>
        <tr>
          <td><strong>Quarantined unsafe</strong></td>
          <td>
            Unsafe code is wrapped in typed, safe APIs with a small surface area.
          </td>
        </tr>
        <tr>
          <td><strong>API stability</strong></td>
          <td>
            Public interfaces are narrow, versioned, and testable.
          </td>
        </tr>
        <tr>
          <td><strong>Resilience by default</strong></td>
          <td>
            Cancellation, timeouts, backpressure, bounded memory.
          </td>
        </tr>
        <tr>
          <td><strong>Performance is a feature</strong></td>
          <td>
            Avoid unnecessary copies, allow zero/low-copy IO (mmap), expose
            tuning knobs.
          </td>
        </tr>
      </tbody>
    </table>

    <h2>Crate Layout</h2>
    <div class="arch-diagram">
{`llama.rs/
  crates/
    `}<span class="highlight">llama-engine/</span>{`        # narrow-waist trait + streaming API
    llama-models/        # model architectures (Llama/Qwen/Mistral)
    `}<span class="highlight">llama-runtime/</span>{`       # execution: oxidizedMLX, backend selection
    llama-tokenizer/     # tokenizers + chat templates
    llama-sampling/      # samplers + penalties + stop conditions
    llama-kv/            # KV cache layouts + paging/eviction`}
    </div>
    <p>
      Future crates (not yet scaffolded):
    </p>
    <table>
      <thead>
        <tr>
          <th>Crate</th>
          <th>Purpose</th>
          <th>Story</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>llama-server</code></td>
          <td>OpenAI-compatible HTTP API</td>
          <td>LLAMA-009</td>
        </tr>
        <tr>
          <td><code>llama-cli</code></td>
          <td>CLI runner for debug + bench</td>
          <td>Milestone A</td>
        </tr>
        <tr>
          <td><code>llama-rag</code></td>
          <td>oxidizedRAG adapter</td>
          <td>LLAMA-011</td>
        </tr>
        <tr>
          <td><code>llama-agents</code></td>
          <td>oxidizedgraph agent nodes</td>
          <td>LLAMA-012</td>
        </tr>
      </tbody>
    </table>

    <h2>The Narrow Waist: LlamaEngine</h2>
    <p>
      Everything plugs into the <code>LlamaEngine</code> trait, defined in{" "}
      <code>llama-engine</code>:
    </p>
    <pre>
      <code>
        <span class="kw">pub trait</span>{" "}<span class="ty">LlamaEngine</span>{": Send + Sync {\n"}
        {"    "}<span class="kw">fn</span>{" "}<span class="fn">load_model</span>{"(&self, spec: &ModelSpec) -> Result<ModelHandle>;\n"}
        {"    "}<span class="kw">fn</span>{" "}<span class="fn">tokenize</span>{"(&self, text: &str) -> Result<Vec<i32>>;\n"}
        {"    "}<span class="kw">fn</span>{" "}<span class="fn">detokenize</span>{"(&self, tokens: &[i32]) -> Result<String>;\n"}
        {"    "}<span class="kw">fn</span>{" "}<span class="fn">prefill</span>{"(&self, session: &mut Session, tokens: &[i32]) -> Result<PrefillResult>;\n"}
        {"    "}<span class="kw">fn</span>{" "}<span class="fn">decode</span>{"(&self, session: &mut Session) -> Result<TokenStream>;\n"}
        {"    "}<span class="kw">fn</span>{" "}<span class="fn">embed</span>{"(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;\n"}
        {"}\n"}
      </code>
    </pre>
    <p>
      Methods take <code>&self</code> (not <code>&mut self</code>) to allow
      shared access across multiple sessions and enable concurrent inference.
      Backends are responsible for their own internal synchronization.
    </p>

    <h2>Acceleration Strategy</h2>
    <p>
      The runtime exposes capability discovery rather than hard-coding platform
      rules:
    </p>
    <pre>
      <code>
        <span class="cm">{"// Feature gates control backend compilation\n"}</span>
        <span class="kw">{"#[cfg(feature = "}</span><span class="st">{'"cpu"'}</span><span class="kw">{")]"}</span>{"\n"}
        <span class="ty">{"Cpu"}</span>{",\n"}
        <span class="kw">{"#[cfg(feature = "}</span><span class="st">{'"metal"'}</span><span class="kw">{")]"}</span>{"\n"}
        <span class="ty">{"Metal"}</span>{",\n\n"}
        <span class="cm">{"// Kernel availability matrix validates op support at startup\n"}</span>
        <span class="kw">let</span>{" matrix = KernelMatrix::probe();\n"}
        <span class="kw">let</span>{" selector = BackendSelector::auto()?;\n"}
      </code>
    </pre>
    <div class="callout">
      <span class="callout-icon">&#x1F6E1;</span>
      <div>
        <strong>Backend parity gate:</strong> Metal cannot become the default
        backend unless parity tests pass against the CPU reference
        implementation.
      </div>
    </div>

    <h2>Session Lifecycle</h2>
    <p>
      Sessions hold runtime state (KV cache, token history) that persists
      across prefill and decode phases. Multiple sessions can exist
      simultaneously with independent state.
    </p>
    <table>
      <thead>
        <tr>
          <th>Property</th>
          <th>Design</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Cancellation</td>
          <td>Cooperative. Dropping a <code>TokenStream</code> terminates graph execution.</td>
        </tr>
        <tr>
          <td>Memory</td>
          <td>Bounded. Context size and KV cache behavior are explicit and configurable.</td>
        </tr>
        <tr>
          <td>Lifecycle</td>
          <td>Session ID maps to KV cache lifecycle (freed or archived on completion).</td>
        </tr>
        <tr>
          <td>Cloning</td>
          <td>Intentionally not <code>Clone</code> &mdash; duplicating KV cache state is not cheap.</td>
        </tr>
      </tbody>
    </table>
  </Layout>
);
