import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function GettingStarted() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">Getting Started</h1>
      <p className="text-lg text-zinc-400 mb-10">
        Install llama.rs crates from crates.io and run your first inference in
        under five minutes.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Prerequisites</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        You need Rust 1.80+ and Cargo. If you're using the workspace from source,
        you'll also want <code className="font-mono text-orange-300/90 text-[13px]">just</code> for the task runner.
      </p>
      <CodeBlock>{`# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install just (task runner)
cargo install just`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Add to Your Project</h2>
      <p className="text-zinc-400 text-[15px] mb-4">
        Add the crates you need to your Cargo.toml:
      </p>
      <CodeBlock title="Cargo.toml">{`[dependencies]
llama-engine    = "0.1"
llama-runtime   = "0.1"
llama-tokenizer = "0.1"
llama-sampling  = "0.1"
llama-models    = "0.1"
llama-kv        = "0.1"`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Quick Example</h2>
      <CodeBlock title="main.rs">{`use llama_tokenizer::{Tokenizer, WhitespaceTokenizer};
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
);`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">From Source</h2>
      <CodeBlock>{`git clone https://github.com/stevedores-org/llama.rs
cd llama.rs

# Run all checks (format, clippy, test)
just ci

# Or individually:
just fmt        # format code
just clippy     # lint
just test       # run tests
just check      # type-check`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">CLI Demo</h2>
      <CodeBlock>{`cargo run -p llama-cli -- generate \\
  --prompt "hello world" \\
  --max-tokens 16 \\
  --temperature 1.0 \\
  --seed 42`}</CodeBlock>

      <Callout icon="💡">
        The demo model uses randomly initialized weights and byte-value decoding.
        It's for testing the pipeline, not generating useful text. Real weight
        loading (safetensors) is planned for Milestone C.
      </Callout>
    </Layout>
  );
}
