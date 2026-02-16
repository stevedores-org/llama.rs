import Layout from "@/components/Layout";
import StatusBadge from "@/components/StatusBadge";

function StoryTable({
  stories,
}: {
  stories: { id: string; issue: string; description: string; status: "done" | "wip" | "planned" }[];
}) {
  return (
    <div className="border border-zinc-800 rounded-xl overflow-hidden my-4">
      <table className="w-full text-[13px]">
        <thead>
          <tr className="border-b border-zinc-800 bg-zinc-900/50">
            <th className="text-left px-4 py-2.5 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Story</th>
            <th className="text-left px-4 py-2.5 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Issue</th>
            <th className="text-left px-4 py-2.5 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Description</th>
            <th className="text-left px-4 py-2.5 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Status</th>
          </tr>
        </thead>
        <tbody className="text-zinc-400">
          {stories.map((s) => (
            <tr key={s.id} className="border-b border-zinc-800/60 last:border-0">
              <td className="px-4 py-2.5 text-zinc-300 font-mono text-xs">{s.id}</td>
              <td className="px-4 py-2.5">
                <a href={`https://github.com/stevedores-org/llama.rs/issues/${s.issue}`} className="text-orange-400 hover:text-orange-300">
                  #{s.issue}
                </a>
              </td>
              <td className="px-4 py-2.5">{s.description}</td>
              <td className="px-4 py-2.5"><StatusBadge status={s.status} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Roadmap() {
  return (
    <Layout>
      <h1 className="text-3xl font-extrabold tracking-tight mb-2">Roadmap</h1>
      <p className="text-lg text-zinc-400 mb-10">
        Five milestones from "Hello Inference" to RAG + Agents. Twelve user stories across five epics.
      </p>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-4 pb-2 border-b border-zinc-800/60">Milestones</h2>
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900/50">
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Milestone</th>
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Goal</th>
              <th className="text-left px-5 py-3 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">Status</th>
            </tr>
          </thead>
          <tbody className="text-zinc-400">
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">A — Hello Inference</td><td className="px-5 py-3">Tiny model, CPU, greedy sampling, CLI generate</td><td className="px-5 py-3"><StatusBadge status="wip" /></td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">B — KV Cache</td><td className="px-5 py-3">Prefill + decode KV equivalence, streaming</td><td className="px-5 py-3"><StatusBadge status="wip" /></td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">C — Weight Loading</td><td className="px-5 py-3">Safetensors, tensor mapping, memory checks</td><td className="px-5 py-3"><StatusBadge status="planned" /></td></tr>
            <tr className="border-b border-zinc-800/60"><td className="px-5 py-3 text-zinc-200 font-medium">D — Metal</td><td className="px-5 py-3">Op parity suite, Metal behind parity gate</td><td className="px-5 py-3"><StatusBadge status="planned" /></td></tr>
            <tr><td className="px-5 py-3 text-zinc-200 font-medium">E — RAG + Agents</td><td className="px-5 py-3">oxidizedRAG adapter, oxidizedgraph agent nodes</td><td className="px-5 py-3"><StatusBadge status="planned" /></td></tr>
          </tbody>
        </table>
      </div>

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-2 pb-2 border-b border-zinc-800/60">Epic 1: Repository Foundation</h2>
      <p className="text-zinc-400 text-[15px] mb-2">Establish the workspace, define core traits, implement tokenization.</p>
      <StoryTable stories={[
        { id: "LLAMA-001", issue: "4", description: "Modular workspace and narrow-waist LlamaEngine trait", status: "done" },
        { id: "LLAMA-002", issue: "5", description: "Robust tokenizer with streaming UTF-8 support", status: "wip" },
        { id: "LLAMA-003", issue: "6", description: "Foundational sampling (greedy, top-k/p, temperature)", status: "done" },
      ]} />

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-2 pb-2 border-b border-zinc-800/60">Epic 2: Inference & KV Cache</h2>
      <p className="text-zinc-400 text-[15px] mb-2">Graph-friendly model architectures and memory management.</p>
      <StoryTable stories={[
        { id: "LLAMA-004", issue: "7", description: "First-class KV Cache (prefill, decode, paging)", status: "done" },
        { id: "LLAMA-005", issue: "8", description: "Llama 3 and Qwen model blocks in Rust", status: "wip" },
        { id: "LLAMA-006", issue: "9", description: "Prefill vs. Decode verification suite", status: "done" },
      ]} />

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-2 pb-2 border-b border-zinc-800/60">Epic 3: Hardware Acceleration</h2>
      <p className="text-zinc-400 text-[15px] mb-2">Integrate oxidizedMLX and enforce CPU/Metal parity.</p>
      <StoryTable stories={[
        { id: "LLAMA-007", issue: "10", description: "Runtime selector for oxidizedMLX backends", status: "done" },
        { id: "LLAMA-008", issue: "11", description: "Backend Parity Gate (Metal vs CPU golden tests)", status: "planned" },
      ]} />

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-2 pb-2 border-b border-zinc-800/60">Epic 4: Application Surface</h2>
      <p className="text-zinc-400 text-[15px] mb-2">HTTP server with streaming, sessions, and cancellation.</p>
      <StoryTable stories={[
        { id: "LLAMA-009", issue: "12", description: "OpenAI-compatible HTTP server", status: "planned" },
        { id: "LLAMA-010", issue: "13", description: "Session and cancellation management", status: "planned" },
      ]} />

      <h2 className="text-xl font-bold tracking-tight mt-10 mb-2 pb-2 border-b border-zinc-800/60">Epic 5: Agentic Compute</h2>
      <p className="text-zinc-400 text-[15px] mb-2">RAG and multi-agent orchestration via the oxidized ecosystem.</p>
      <StoryTable stories={[
        { id: "LLAMA-011", issue: "14", description: "RAG adapter for oxidizedRAG", status: "planned" },
        { id: "LLAMA-012", issue: "15", description: "Agent orchestration via oxidizedgraph", status: "planned" },
      ]} />
    </Layout>
  );
}
