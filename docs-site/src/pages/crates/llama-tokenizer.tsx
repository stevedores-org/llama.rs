import type { FC } from "hono/jsx";
import { Layout } from "../../components/Layout";

export const LlamaTokenizerPage: FC = () => (
  <Layout title="llama-tokenizer" activePath="/crates/llama-tokenizer">
    <h1>llama-tokenizer</h1>
    <p class="lead">
      Deterministic text-to-token conversion with streaming UTF-8 decoding,
      pluggable backends, and asset loading for HuggingFace and SentencePiece
      vocabularies.
    </p>
    <pre><code>cargo add llama-tokenizer</code></pre>

    <h2>Tokenizer Trait</h2>
    <pre>
      <code>
        <span class="kw">pub trait</span>{" "}<span class="ty">Tokenizer</span>{": Send + Sync {\n"}
        {"    "}<span class="kw">fn</span>{" "}<span class="fn">encode</span>{"(&self, text: &str) -> TokenizerResult<Vec<i32>>;\n"}
        {"    "}<span class="kw">fn</span>{" "}<span class="fn">decode</span>{"(&self, tokens: &[i32]) -> TokenizerResult<String>;\n"}
        {"    "}<span class="kw">fn</span>{" "}<span class="fn">decode_token</span>{"(&self, token: i32, state: &mut DecodingState) -> TokenizerResult<String>;\n"}
        {"    "}<span class="kw">fn</span>{" "}<span class="fn">vocab_size</span>{"(&self) -> usize;\n"}
        {"}\n"}
      </code>
    </pre>

    <h2>Implementations</h2>
    <h3>WhitespaceTokenizer</h3>
    <p>
      Reference implementation for testing. Splits on whitespace, assigns
      sequential IDs. Thread-safe via <code>RwLock</code> for the vocabulary
      map.
    </p>
    <pre>
      <code>
        <span class="kw">let</span>{" tok = WhitespaceTokenizer::new();\n"}
        <span class="kw">let</span>{" ids = tok.encode("}<span class="st">{'"hello world"'}</span>{").unwrap();\n"}
        <span class="kw">let</span>{" text = tok.decode(&ids).unwrap();\n"}
        <span class="mc">assert_eq!</span>{"(text, "}<span class="st">{'"hello world"'}</span>{");\n"}
      </code>
    </pre>

    <h3>LoadedTokenizer</h3>
    <p>
      Production tokenizer that loads vocabularies from asset files:
    </p>
    <table>
      <thead>
        <tr><th>Format</th><th>Method</th></tr>
      </thead>
      <tbody>
        <tr><td>HuggingFace <code>tokenizer.json</code></td><td><code>load_hf_vocab(path)</code></td></tr>
        <tr><td>SentencePiece <code>.model</code></td><td><code>load_sentencepiece_vocab(path)</code></td></tr>
      </tbody>
    </table>
    <p>
      Uses greedy longest-match subword encoding and concatenative decoding.
    </p>

    <h2>Streaming Decoding</h2>
    <p>
      The <code>DecodingState</code> struct handles partial UTF-8 sequences
      during streaming token-by-token decoding:
    </p>
    <pre>
      <code>
        <span class="kw">let mut</span>{" state = DecodingState::new();\n\n"}
        <span class="kw">for</span>{" token "}<span class="kw">in</span>{" token_stream {\n"}
        {"    "}<span class="kw">let</span>{" chunk = tokenizer.decode_token(token, &mut state)?;\n"}
        {"    print!("}<span class="st">{'"{}",chunk'}</span>{"); "}<span class="cm">{"// emit immediately\n"}</span>
        {"}\n\n"}
        <span class="cm">{"// Verify no pending bytes at end of stream\n"}</span>
        {"state.finalize()?;\n"}
      </code>
    </pre>

    <h2>Error Handling</h2>
    <table>
      <thead>
        <tr><th>Variant</th><th>When</th></tr>
      </thead>
      <tbody>
        <tr><td><code>InvalidToken(i32)</code></td><td>Token ID not in vocabulary</td></tr>
        <tr><td><code>EncodingError(String)</code></td><td>Text cannot be encoded</td></tr>
        <tr><td><code>DecodingError(String)</code></td><td>Incomplete UTF-8 at end of stream</td></tr>
      </tbody>
    </table>

    <div class="callout">
      <span class="callout-icon">&#x1F517;</span>
      <div>
        <a href="https://crates.io/crates/llama-tokenizer">crates.io</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/tree/main/crates/llama-tokenizer">source</a>{" | "}
        <a href="https://github.com/stevedores-org/llama.rs/issues/5">LLAMA-002</a>
      </div>
    </div>
  </Layout>
);
