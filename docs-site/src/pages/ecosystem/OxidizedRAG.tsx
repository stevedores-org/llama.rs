import Layout from "@/components/Layout";
import Callout from "@/components/Callout";
import { Link } from "react-router-dom";

export default function OxidizedRAG() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">oxidizedRAG</h1>
      <p className="text-lg text-zinc-400 mb-6">
        Retrieval-Augmented Generation toolkit in Rust. Vector search, document chunking,
        and citation-aware prompting.
      </p>
      <p className="text-zinc-400 text-[15px] mb-4">
        oxidizedRAG provides the retrieval and memory layer for context-augmented
        generation. It handles document ingestion, embedding, vector similarity
        search, and citation tracking.
      </p>
      <Callout icon="🔌">
        <strong className="text-zinc-100">llama.rs integration:</strong>{" "}
        The planned llama-rag crate (LLAMA-011) will adapt oxidizedRAG for use
        with the LlamaEngine trait, enabling retrieval-augmented inference pipelines.
      </Callout>
      <div className="flex gap-3 mt-8">
        <a href="https://github.com/stevedores-org/oxidizedRAG" className="bg-orange-500 hover:bg-orange-600 text-black font-semibold px-5 py-2.5 rounded-lg transition text-sm">
          View on GitHub
        </a>
        <Link to="/architecture" className="border border-zinc-700 hover:border-zinc-500 px-5 py-2.5 rounded-lg transition text-sm text-zinc-300">
          Architecture
        </Link>
      </div>
    </Layout>
  );
}
