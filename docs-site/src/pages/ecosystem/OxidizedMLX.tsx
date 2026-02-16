import Layout from "@/components/Layout";
import Callout from "@/components/Callout";
import { Link } from "react-router-dom";

export default function OxidizedMLX() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">oxidizedMLX</h1>
      <p className="text-lg text-zinc-400 mb-6">
        Rust bindings for MLX — Apple's array framework for machine learning on Apple silicon.
      </p>
      <p className="text-zinc-400 text-[15px] mb-4">
        oxidizedMLX provides the tensor runtime and Metal GPU acceleration that
        powers llama.rs inference. It handles memory management, kernel dispatch,
        and device-to-host transfers for the Metal backend.
      </p>
      <Callout icon="🔌">
        <strong className="text-zinc-100">llama.rs integration:</strong>{" "}
        llama-runtime uses oxidizedMLX for Metal backend acceleration. The CPU
        backend is pure Rust. Backend selection is automatic via the KernelMatrix probe.
      </Callout>
      <div className="flex gap-3 mt-8">
        <a href="https://github.com/stevedores-org/oxidizedMLX" className="bg-orange-500 hover:bg-orange-600 text-black font-semibold px-5 py-2.5 rounded-lg transition text-sm">
          View on GitHub
        </a>
        <Link to="/architecture" className="border border-zinc-700 hover:border-zinc-500 px-5 py-2.5 rounded-lg transition text-sm text-zinc-300">
          Architecture
        </Link>
      </div>
    </Layout>
  );
}
