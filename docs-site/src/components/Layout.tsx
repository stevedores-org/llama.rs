import type { FC, PropsWithChildren } from "hono/jsx";
import { css } from "../styles/theme";

const BASE = "/llama-rs";

interface NavItem {
  label: string;
  href: string;
  icon: string;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const navigation: NavSection[] = [
  {
    title: "Overview",
    items: [
      { label: "Introduction", href: "/", icon: "\u25b8" },
      { label: "Getting Started", href: "/getting-started", icon: "\u25b8" },
      { label: "Architecture", href: "/architecture", icon: "\u25b8" },
      { label: "Roadmap", href: "/roadmap", icon: "\u25b8" },
    ],
  },
  {
    title: "Crates",
    items: [
      { label: "llama-engine", href: "/crates/llama-engine", icon: "\u25cb" },
      { label: "llama-tokenizer", href: "/crates/llama-tokenizer", icon: "\u25cb" },
      { label: "llama-models", href: "/crates/llama-models", icon: "\u25cb" },
      { label: "llama-sampling", href: "/crates/llama-sampling", icon: "\u25cb" },
      { label: "llama-kv", href: "/crates/llama-kv", icon: "\u25cb" },
      { label: "llama-runtime", href: "/crates/llama-runtime", icon: "\u25cb" },
    ],
  },
  {
    title: "Ecosystem",
    items: [
      { label: "oxidizedMLX", href: "/ecosystem/oxidized-mlx", icon: "\u2666" },
      { label: "oxidizedRAG", href: "/ecosystem/oxidized-rag", icon: "\u2666" },
      { label: "oxidizedgraph", href: "/ecosystem/oxidized-graph", icon: "\u2666" },
    ],
  },
];

interface LayoutProps extends PropsWithChildren {
  title: string;
  activePath: string;
}

export const Layout: FC<LayoutProps> = ({ title, activePath, children }) => {
  return (
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{title} &mdash; llama.rs docs</title>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
        <style dangerouslySetInnerHTML={{ __html: css }} />
      </head>
      <body>
        <div class="layout">
          <nav class="sidebar">
            <div class="sidebar-header">
              <a href={`${BASE}/`} class="sidebar-logo">
                <span class="dot">&gt;_</span> llama.rs
                <span class="sidebar-badge">v0.1.1</span>
              </a>
            </div>
            <div class="sidebar-nav">
              {navigation.map((section) => (
                <>
                  <div class="nav-section">{section.title}</div>
                  {section.items.map((item) => (
                    <a
                      href={`${BASE}${item.href}`}
                      class={`nav-link${activePath === item.href ? " active" : ""}`}
                    >
                      <span class="nav-icon">{item.icon}</span>
                      {item.label}
                    </a>
                  ))}
                </>
              ))}
            </div>
            <div class="sidebar-footer">
              <a href="https://github.com/stevedores-org/llama.rs">GitHub</a>
              {" \u00b7 "}
              <a href="https://crates.io/crates/llama-engine">crates.io</a>
              {" \u00b7 "}
              <a href="https://docs.stevedores.org">stevedores.org</a>
            </div>
          </nav>
          <main class="main">
            <div class="content">{children}</div>
          </main>
        </div>
      </body>
    </html>
  );
};
