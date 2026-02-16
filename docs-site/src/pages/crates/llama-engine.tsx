import type { FC } from "hono/jsx";
import { Layout } from "../../components/Layout";

export const LlamaEnginePage: FC = () => (
  <Layout title="llama-engine" activePath="/crates/llama-engine">
    <h1>llama-engine</h1>
    <p class="lead">
      The "narrow waist" of the llama.rs stack. Defines the core{" "}
      <code>LlamaEngine</code> trait and associated types that all other crates
      depend on.
    </p>
    <pre><code>cargo add llama-engine</code></pre>

    <h2>Overview</h2>
    <p>
      <code>llama-engine</code> provides the single stable interface that
      consumers (applications, servers, agents) program against. Backend
      implementations can swap CPU, Metal, or FFI runtimes without changing
      application code.
    </p>

    <h2>Core Trait</h2>
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
      shared access across multiple sessions and concurrent inference. Backends
      use interior mutability for any necessary synchronization.
    </p>

    <h2>Key Types</h2>
    <table>
      <thead>
        <tr><th>Type</th><th>Description</th></tr>
      </thead>
      <tbody>
        <tr><td><code>TokenId</code></td><td>Alias for <code>i32</code> (FFI compatible). Logically non-negative.</td></tr>
        <tr><td><code>ModelSpec</code></td><td>Configuration for loading a model (path, context size).</td></tr>
        <tr><td><code>ModelHandle</code></td><td>Opaque handle to a loaded model. No public fields.</td></tr>
        <tr><td><code>Session</code></td><td>Active inference session with KV cache state. Not <code>Clone</code>.</td></tr>
        <tr><td><code>PrefillResult</code></td><td>Result of the prefill phase (marked <code>#[must_use]</code>).</td></tr>
        <tr><td><code>LlamaError</code></td><td>Typed error enum: <code>ModelLoad</code>, <code>Tokenization</code>, <code>Inference</code>.</td></tr>
      </tbody>
    </table>

    <h2>Design Notes</h2>
    <h3>Interior Mutability</h3>
    <p>
      <code>LlamaEngine</code> takes <code>&self</code> so it can be shared
      behind an <code>Arc</code>. Backends using <code>Mutex</code> or{" "}
      <code>RwLock</code> internally are responsible for their own
      synchronization.
    </p>
    <h3>Session Identity</h3>
    <p>
      Each <code>Session</code> has a private <code>uuid::Uuid</code> ID. This
      is intentionally hidden so backends can rely on it as a stable key for KV
      cache lifecycle management.
    </p>

    <div class="callout">
      <span class="callout-icon">&#x1F517;</span>
      <div>
        <a href="https://crates.io/crates/llama-engine">crates.io</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-engine">source</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/4">LLAMA-001</a>
      </div>
    </div>
  </Layout>
);
