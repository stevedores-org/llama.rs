import Layout from "@/components/Layout";
import Callout from "@/components/Callout";
import { Link } from "react-router-dom";

export default function OxidizedGraph() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">oxidizedgraph</h1>
      <p className="text-lg text-zinc-400 mb-6">
        Directed graph execution engine for multi-agent orchestration in Rust.
      </p>
      <p className="text-zinc-400 text-[15px] mb-4">
        oxidizedgraph provides a typed, async graph execution runtime for building
        agent pipelines. Nodes can be inference calls, retrieval steps, tool
        invocations, or custom logic.
      </p>
      <Callout icon="🔌">
        <strong className="text-zinc-100">llama.rs integration:</strong>{" "}
        The planned llama-agents crate (LLAMA-012) will provide oxidizedgraph
        nodes for: Retrieve, Rerank, Prompt, and Generate — enabling end-to-end
        agent workflows.
      </Callout>
      <div className="flex gap-3 mt-8">
        <a href="https://github.com/stevedores-org/oxidizedgraph" className="bg-orange-500 hover:bg-orange-600 text-black font-semibold px-5 py-2.5 rounded-lg transition text-sm">
          View on GitHub
        </a>
        <Link to="/architecture" className="border border-zinc-700 hover:border-zinc-500 px-5 py-2.5 rounded-lg transition text-sm text-zinc-300">
          Architecture
        </Link>
      </div>
    </Layout>
  );
}
