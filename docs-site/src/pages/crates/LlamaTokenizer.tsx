import Layout from "@/components/Layout";
import CodeBlock from "@/components/CodeBlock";
import Callout from "@/components/Callout";

export default function LlamaTokenizer() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">llama-tokenizer</h1>
      <p className="text-lg text-zinc-400 mb-6">
        Deterministic text-to-token conversion with streaming UTF-8 decoding and pluggable backends.
      </p>
      <CodeBlock>cargo add llama-tokenizer</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Tokenizer Trait</h2>
      <CodeBlock>{`pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> TokenizerResult<Vec<i32>>;
    fn decode(&self, tokens: &[i32]) -> TokenizerResult<String>;
    fn decode_token(&self, token: i32, state: &mut DecodingState) -> TokenizerResult<String>;
    fn vocab_size(&self) -> usize;
}`}</CodeBlock>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Implementations</h2>
      <h3 className="text-base font-semibold mt-6 mb-2 text-zinc-200">WhitespaceTokenizer</h3>
      <p className="text-zinc-400 text-[15px] mb-4">
        Reference implementation for testing. Splits on whitespace, assigns sequential IDs.
        Thread-safe via RwLock.
      </p>
      <CodeBlock>{`let tok = WhitespaceTokenizer::new();
let ids = tok.encode("hello world").unwrap();
let text = tok.decode(&ids).unwrap();
assert_eq!(text, "hello world");`}</CodeBlock>

      <h3 className="text-base font-semibold mt-6 mb-2 text-zinc-200">LoadedTokenizer</h3>
      <p className="text-zinc-400 text-[15px] mb-4">
        Production tokenizer that loads vocabularies from HuggingFace{" "}
        <code className="font-mono text-orange-300/90 text-[13px]">tokenizer.json</code>{" "}
        or SentencePiece assets. Greedy longest-match subword encoding.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-3 pb-2 border-b border-zinc-800/60">Streaming Decoding</h2>
      <CodeBlock>{`let mut state = DecodingState::new();

for token in token_stream {
    let chunk = tokenizer.decode_token(token, &mut state)?;
    print!("{}", chunk); // emit immediately
}

// Verify no pending bytes at end of stream
state.finalize()?;`}</CodeBlock>

      <Callout icon="🔗">
        <a href="https://crates.io/crates/llama-tokenizer" className="text-orange-400 hover:text-orange-300">crates.io</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-tokenizer" className="text-orange-400 hover:text-orange-300">source</a>{" · "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/5" className="text-orange-400 hover:text-orange-300">LLAMA-002</a>
      </Callout>
    </Layout>
  );
}
