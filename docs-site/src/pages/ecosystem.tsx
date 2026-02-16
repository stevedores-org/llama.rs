import type { FC } from "hono/jsx";
import { Layout } from "../components/Layout";

const BASE = "/llama-rs";

interface EcosystemPageProps {
  activePath: string;
  name: string;
  repo: string;
  tagline: string;
  description: string;
  integrationNote: string;
}

const EcosystemContent: FC<EcosystemPageProps> = ({
  activePath,
  name,
  repo,
  tagline,
  description,
  integrationNote,
}) => (
  <Layout title={name} activePath={activePath}>
    <h1>{name}</h1>
    <p class="lead">{tagline}</p>
    <p>{description}</p>
    <div class="callout">
      <span class="callout-icon">&#x1F50C;</span>
      <div>
        <strong>llama.rs integration:</strong> {integrationNote}
      </div>
    </div>
    <div class="hero-actions" style="margin-top: 24px">
      <a href={`https://github.com/stevedores-org/${repo}`} class="btn btn-primary">
        View on GitHub
      </a>
      <a href={`${BASE}/architecture`} class="btn btn-ghost">
        Architecture
      </a>
    </div>
  </Layout>
);

export const OxidizedMLXPage: FC = () => (
  <EcosystemContent
    activePath="/ecosystem/oxidized-mlx"
    name="oxidizedMLX"
    repo="oxidizedMLX"
    tagline="Rust bindings for MLX — Apple's array framework for machine learning on Apple silicon."
    description="oxidizedMLX provides the tensor runtime and Metal GPU acceleration that powers llama.rs inference. It handles memory management, kernel dispatch, and device-to-host transfers for the Metal backend."
    integrationNote="llama-runtime uses oxidizedMLX for Metal backend acceleration. The CPU backend is pure Rust. Backend selection is automatic via the KernelMatrix probe."
  />
);

export const OxidizedRAGPage: FC = () => (
  <EcosystemContent
    activePath="/ecosystem/oxidized-rag"
    name="oxidizedRAG"
    repo="oxidizedRAG"
    tagline="Retrieval-Augmented Generation toolkit in Rust. Vector search, document chunking, and citation-aware prompting."
    description="oxidizedRAG provides the retrieval and memory layer for context-augmented generation. It handles document ingestion, embedding, vector similarity search, and citation tracking."
    integrationNote="The planned llama-rag crate (LLAMA-011) will adapt oxidizedRAG for use with the LlamaEngine trait, enabling retrieval-augmented inference pipelines."
  />
);

export const OxidizedGraphPage: FC = () => (
  <EcosystemContent
    activePath="/ecosystem/oxidized-graph"
    name="oxidizedgraph"
    repo="oxidizedgraph"
    tagline="Directed graph execution engine for multi-agent orchestration in Rust."
    description="oxidizedgraph provides a typed, async graph execution runtime for building agent pipelines. Nodes can be inference calls, retrieval steps, tool invocations, or custom logic."
    integrationNote="The planned llama-agents crate (LLAMA-012) will provide oxidizedgraph nodes for: Retrieve, Rerank, Prompt, and Generate — enabling end-to-end agent workflows."
  />
);
