import { Hono } from "hono";
import { IndexPage } from "./pages/index";
import { GettingStartedPage } from "./pages/getting-started";
import { ArchitecturePage } from "./pages/architecture";
import { RoadmapPage } from "./pages/roadmap";
import { LlamaEnginePage } from "./pages/crates/llama-engine";
import { LlamaTokenizerPage } from "./pages/crates/llama-tokenizer";
import { LlamaModelsPage } from "./pages/crates/llama-models";
import { LlamaSamplingPage } from "./pages/crates/llama-sampling";
import { LlamaKvPage } from "./pages/crates/llama-kv";
import { LlamaRuntimePage } from "./pages/crates/llama-runtime";
import {
  OxidizedMLXPage,
  OxidizedRAGPage,
  OxidizedGraphPage,
} from "./pages/ecosystem";

const BASE = "/llama-rs";

export const app = new Hono();

const page = (Component: () => any) => (c: any) =>
  c.html("<!DOCTYPE html>" + Component());

app.get(`${BASE}/`, page(IndexPage));
app.get(`${BASE}/getting-started`, page(GettingStartedPage));
app.get(`${BASE}/architecture`, page(ArchitecturePage));
app.get(`${BASE}/roadmap`, page(RoadmapPage));

app.get(`${BASE}/crates/llama-engine`, page(LlamaEnginePage));
app.get(`${BASE}/crates/llama-tokenizer`, page(LlamaTokenizerPage));
app.get(`${BASE}/crates/llama-models`, page(LlamaModelsPage));
app.get(`${BASE}/crates/llama-sampling`, page(LlamaSamplingPage));
app.get(`${BASE}/crates/llama-kv`, page(LlamaKvPage));
app.get(`${BASE}/crates/llama-runtime`, page(LlamaRuntimePage));

app.get(`${BASE}/ecosystem/oxidized-mlx`, page(OxidizedMLXPage));
app.get(`${BASE}/ecosystem/oxidized-rag`, page(OxidizedRAGPage));
app.get(`${BASE}/ecosystem/oxidized-graph`, page(OxidizedGraphPage));

// Redirect bare /llama-rs to /llama-rs/
app.get("/llama-rs", (c) => c.redirect(`${BASE}/`));

export const routes = [
  "/",
  "/getting-started",
  "/architecture",
  "/roadmap",
  "/crates/llama-engine",
  "/crates/llama-tokenizer",
  "/crates/llama-models",
  "/crates/llama-sampling",
  "/crates/llama-kv",
  "/crates/llama-runtime",
  "/ecosystem/oxidized-mlx",
  "/ecosystem/oxidized-rag",
  "/ecosystem/oxidized-graph",
];
