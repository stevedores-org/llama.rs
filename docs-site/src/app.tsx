import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import GettingStarted from "./pages/GettingStarted";
import Architecture from "./pages/Architecture";
import Roadmap from "./pages/Roadmap";
import LlamaEngine from "./pages/crates/LlamaEngine";
import LlamaTokenizer from "./pages/crates/LlamaTokenizer";
import LlamaModels from "./pages/crates/LlamaModels";
import LlamaSampling from "./pages/crates/LlamaSampling";
import LlamaKv from "./pages/crates/LlamaKv";
import LlamaRuntime from "./pages/crates/LlamaRuntime";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/getting-started" element={<GettingStarted />} />
      <Route path="/architecture" element={<Architecture />} />
      <Route path="/roadmap" element={<Roadmap />} />
      <Route path="/crates/llama-engine" element={<LlamaEngine />} />
      <Route path="/crates/llama-tokenizer" element={<LlamaTokenizer />} />
      <Route path="/crates/llama-models" element={<LlamaModels />} />
      <Route path="/crates/llama-sampling" element={<LlamaSampling />} />
      <Route path="/crates/llama-kv" element={<LlamaKv />} />
      <Route path="/crates/llama-runtime" element={<LlamaRuntime />} />
    </Routes>
  );
}
