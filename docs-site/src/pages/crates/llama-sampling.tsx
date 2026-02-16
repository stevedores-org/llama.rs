import type { FC } from "hono/jsx";
import { Layout } from "../../components/Layout";

export const LlamaSamplingPage: FC = () => (
  <Layout title="llama-sampling" activePath="/crates/llama-sampling">
    <h1>llama-sampling</h1>
    <p class="lead">
      Sampling and decoding strategies with deterministic seeded RNG for
      reproducible generation.
    </p>
    <pre><code>cargo add llama-sampling</code></pre>

    <h2>Strategies</h2>
    <table>
      <thead>
        <tr><th>Strategy</th><th>Description</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Greedy</strong></td>
          <td>Argmax selection. Always picks the highest-probability token.</td>
        </tr>
        <tr>
          <td><strong>Temperature</strong></td>
          <td>Scale logits by <code>1/T</code> before softmax. Higher T = more random.</td>
        </tr>
        <tr>
          <td><strong>Top-k</strong></td>
          <td>Keep only the top k tokens by probability, zero out the rest.</td>
        </tr>
        <tr>
          <td><strong>Top-p (nucleus)</strong></td>
          <td>Keep the smallest set of tokens whose cumulative probability exceeds p.</td>
        </tr>
        <tr>
          <td><strong>Repetition Penalty</strong></td>
          <td>Penalize tokens that appear in the generation history. Handles negative logits correctly.</td>
        </tr>
      </tbody>
    </table>

    <h2>Usage</h2>
    <pre>
      <code>
        <span class="kw">use</span>{" llama_sampling::{Sampler, SamplingConfig, SamplingStrategy};\n\n"}
        <span class="kw">let mut</span>{" sampler = Sampler::new(SamplingConfig {\n"}
        {"    strategy: SamplingStrategy::Stochastic,\n"}
        {"    temperature: "}<span class="nr">{"0.8"}</span>{",\n"}
        {"    top_k: "}<span class="nr">{"40"}</span>{",\n"}
        {"    top_p: "}<span class="nr">{"0.95"}</span>{",\n"}
        {"    repetition_penalty: "}<span class="nr">{"1.1"}</span>{",\n"}
        {"    seed: "}<span class="nr">{"42"}</span>{",\n"}
        {"    ..SamplingConfig::default()\n"}
        {"})?;\n\n"}
        <span class="cm">{"// Sample a token from logits, with generation history\n"}</span>
        <span class="kw">let</span>{" token = sampler.sample(&logits, &history)?;\n"}
      </code>
    </pre>

    <h2>Configuration</h2>
    <table>
      <thead>
        <tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr>
      </thead>
      <tbody>
        <tr><td><code>strategy</code></td><td><code>SamplingStrategy</code></td><td>Greedy</td><td>Greedy or Stochastic</td></tr>
        <tr><td><code>temperature</code></td><td><code>f32</code></td><td>1.0</td><td>Must be &gt; 0</td></tr>
        <tr><td><code>top_k</code></td><td><code>usize</code></td><td>0 (disabled)</td><td>0 = no top-k filtering</td></tr>
        <tr><td><code>top_p</code></td><td><code>f32</code></td><td>1.0 (disabled)</td><td>1.0 = no top-p filtering</td></tr>
        <tr><td><code>repetition_penalty</code></td><td><code>f32</code></td><td>1.0 (disabled)</td><td>1.0 = no penalty</td></tr>
        <tr><td><code>seed</code></td><td><code>u64</code></td><td>0</td><td>RNG seed (0 auto-maps to 1)</td></tr>
      </tbody>
    </table>

    <h2>Deterministic RNG</h2>
    <p>
      Uses xorshift64 for fast, reproducible random numbers. The RNG state is
      stored in the <code>Sampler</code> and mutated on each{" "}
      <code>sample()</code> call, ensuring same seed + same logits = same
      output.
    </p>
    <pre>
      <code>
        <span class="cm">{"// Same seed always produces same sequence\n"}</span>
        <span class="kw">let mut</span>{" s1 = Sampler::new(config.clone())?;\n"}
        <span class="kw">let mut</span>{" s2 = Sampler::new(config)?;\n"}
        <span class="mc">assert_eq!</span>{"(s1.sample(&logits, &[])?, s2.sample(&logits, &[])?);\n"}
      </code>
    </pre>

    <div class="callout">
      <span class="callout-icon">&#x1F517;</span>
      <div>
        <a href="https://crates.io/crates/llama-sampling">crates.io</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-sampling">source</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/6">LLAMA-003</a>
      </div>
    </div>
  </Layout>
);
