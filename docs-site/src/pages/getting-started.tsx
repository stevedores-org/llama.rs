import type { FC } from "hono/jsx";
import { Layout } from "../components/Layout";

export const GettingStartedPage: FC = () => (
  <Layout title="Getting Started" activePath="/getting-started">
    <h1>Getting Started</h1>
    <p class="lead">
      Install llama.rs crates from crates.io and run your first inference in
      under five minutes.
    </p>

    <h2>Prerequisites</h2>
    <p>
      You need Rust 1.80+ and Cargo. If you're using the workspace from source,
      you'll also want <code>just</code> for the task runner.
    </p>
    <pre>
      <code>{`# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install just (task runner)
cargo install just`}</code>
    </pre>

    <h2>Add to Your Project</h2>
    <p>
      Add the crates you need to your <code>Cargo.toml</code>. Most projects
      will want at minimum the engine trait and a runtime:
    </p>
    <pre>
      <code>{`[dependencies]
llama-engine    = "0.1"
llama-runtime   = "0.1"
llama-tokenizer = "0.1"
llama-sampling  = "0.1"
llama-models    = "0.1"
llama-kv        = "0.1"`}</code>
    </pre>

    <h2>Quick Example</h2>
    <p>
      Here is a minimal generation pipeline using the built-in tiny model and
      whitespace tokenizer (for demo/testing):
    </p>
    <pre><code>{`use llama_tokenizer::{Tokenizer, WhitespaceTokenizer};
use llama_sampling::{Sampler, SamplingConfig, SamplingStrategy};
use llama_kv::{KVLayout, LayerKVCache};

// 1. Tokenize a prompt
let tokenizer = WhitespaceTokenizer::new();
let tokens = tokenizer.encode("hello world").unwrap();

// 2. Configure the sampler
let sampler = Sampler::new(SamplingConfig {
    strategy: SamplingStrategy::Stochastic,
    temperature: 1.0,
    seed: 42,
    ..SamplingConfig::default()
}).unwrap();

// 3. Set up KV cache
let cache = LayerKVCache::new(
    64,  // max_seq_len
    4,   // n_heads
    8,   // head_dim
    KVLayout::BySequence,
);`}</code></pre>

    <h2>From Source</h2>
    <p>Clone the repository and run the full CI suite:</p>
    <pre>
      <code>{`git clone https://github.com/stevedores-org/llama.rs
cd llama.rs

# Run all checks (format, clippy, test)
just ci

# Or individually:
just fmt        # format code
just clippy     # lint
just test       # run tests
just check      # type-check`}</code>
    </pre>

    <h2>CLI Demo</h2>
    <p>
      The workspace includes <code>llama-cli</code>, a demo CLI that generates
      text with a tiny deterministic model:
    </p>
    <pre>
      <code>{`cargo run -p llama-cli -- generate \\
  --prompt "hello world" \\
  --max-tokens 16 \\
  --temperature 1.0 \\
  --seed 42`}</code>
    </pre>
    <div class="callout">
      <span class="callout-icon">&#x1F4A1;</span>
      <div>
        The demo model uses randomly initialized weights and byte-value
        decoding. It's for testing the pipeline, not generating useful text.
        Real weight loading (safetensors) is planned for Milestone C.
      </div>
    </div>

    <h2>What's Next</h2>
    <p>
      Explore the individual crate docs in the sidebar to understand each
      component, or check the{" "}
      <a href="/llama-rs/architecture">Architecture</a> page for how they fit
      together.
    </p>
  </Layout>
);
