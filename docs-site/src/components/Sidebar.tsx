import { Link, useLocation } from "react-router-dom";

interface NavItem {
  label: string;
  href: string;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const navigation: NavSection[] = [
  {
    title: "Overview",
    items: [
      { label: "Introduction", href: "/" },
      { label: "Getting Started", href: "/getting-started" },
      { label: "Architecture", href: "/architecture" },
      { label: "Roadmap", href: "/roadmap" },
    ],
  },
  {
    title: "Crates",
    items: [
      { label: "llama-engine", href: "/crates/llama-engine" },
      { label: "llama-tokenizer", href: "/crates/llama-tokenizer" },
      { label: "llama-models", href: "/crates/llama-models" },
      { label: "llama-sampling", href: "/crates/llama-sampling" },
      { label: "llama-kv", href: "/crates/llama-kv" },
      { label: "llama-runtime", href: "/crates/llama-runtime" },
    ],
  },
  {
    title: "Ecosystem",
    items: [
      { label: "oxidizedMLX", href: "/ecosystem/oxidized-mlx" },
      { label: "oxidizedRAG", href: "/ecosystem/oxidized-rag" },
      { label: "oxidizedgraph", href: "/ecosystem/oxidized-graph" },
    ],
  },
];

export default function Sidebar() {
  const { pathname } = useLocation();

  return (
    <nav className="w-64 shrink-0 bg-zinc-900/60 border-r border-zinc-800 fixed top-0 left-0 bottom-0 overflow-y-auto hidden lg:flex flex-col">
      <div className="px-5 pt-6 pb-4 border-b border-zinc-800/60">
        <Link to="/" className="flex items-center gap-2.5 group">
          <span className="text-orange-500 font-mono font-bold text-lg">&gt;_</span>
          <span className="font-bold text-lg tracking-tight text-zinc-100">
            llama.rs
          </span>
          <span className="text-[10px] font-semibold text-orange-500 bg-orange-500/10 border border-orange-500/20 rounded px-1.5 py-0.5 leading-none">
            v0.1.1
          </span>
        </Link>
      </div>

      <div className="flex-1 py-3">
        {navigation.map((section) => (
          <div key={section.title} className="mb-1">
            <div className="px-5 py-2 text-[11px] font-semibold text-zinc-500 uppercase tracking-widest">
              {section.title}
            </div>
            {section.items.map((item) => {
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  to={item.href}
                  className={`flex items-center gap-2 px-5 py-[7px] text-[13px] border-l-2 transition-all ${
                    active
                      ? "border-orange-500 text-orange-400 bg-orange-500/[0.06] font-medium"
                      : "border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
                  }`}
                >
                  {section.title === "Crates" && (
                    <span className="w-1.5 h-1.5 rounded-full bg-current opacity-50" />
                  )}
                  {item.label}
                </Link>
              );
            })}
          </div>
        ))}
      </div>

      <div className="px-5 py-4 border-t border-zinc-800/60 text-xs text-zinc-500 flex gap-3">
        <a href="https://github.com/stevedores-org/llama.rs" className="hover:text-orange-400 transition">
          GitHub
        </a>
        <span className="text-zinc-700">&middot;</span>
        <a href="https://crates.io/crates/llama-engine" className="hover:text-orange-400 transition">
          crates.io
        </a>
        <span className="text-zinc-700">&middot;</span>
        <a href="https://stevedores.org" className="hover:text-orange-400 transition">
          stevedores.org
        </a>
      </div>
    </nav>
  );
}
