export const css = /* css */ `
:root {
  --bg: #0a0a0b;
  --bg-surface: #111114;
  --bg-surface-hover: #18181c;
  --bg-code: #161619;
  --border: #25252a;
  --border-subtle: #1c1c20;
  --text: #e8e8ec;
  --text-muted: #8b8b96;
  --text-faint: #5f5f6a;
  --accent: #f97316;
  --accent-muted: #c2510f;
  --accent-surface: rgba(249, 115, 22, 0.08);
  --sidebar-w: 260px;
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', 'SF Mono', Menlo, monospace;
}

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

html { font-size: 15px; -webkit-font-smoothing: antialiased; scroll-behavior: smooth; }

body {
  font-family: var(--font-sans);
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
}

a { color: var(--accent); text-decoration: none; transition: color 0.15s; }
a:hover { color: #fb923c; }

/* Layout */
.layout {
  display: flex;
  min-height: 100vh;
}

/* Sidebar */
.sidebar {
  width: var(--sidebar-w);
  background: var(--bg-surface);
  border-right: 1px solid var(--border);
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  overflow-y: auto;
  z-index: 50;
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 24px 20px 16px;
  border-bottom: 1px solid var(--border-subtle);
}

.sidebar-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.02em;
}

.sidebar-logo .dot { color: var(--accent); }

.sidebar-badge {
  display: inline-block;
  font-size: 0.65rem;
  font-weight: 600;
  color: var(--accent);
  background: var(--accent-surface);
  border: 1px solid rgba(249, 115, 22, 0.2);
  border-radius: 4px;
  padding: 1px 6px;
  margin-left: 4px;
  letter-spacing: 0.04em;
}

.sidebar-nav { padding: 12px 0; flex: 1; }

.nav-section {
  padding: 8px 20px 4px;
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 20px;
  font-size: 0.87rem;
  color: var(--text-muted);
  transition: all 0.12s;
  border-left: 2px solid transparent;
}

.nav-link:hover {
  color: var(--text);
  background: var(--bg-surface-hover);
}

.nav-link.active {
  color: var(--accent);
  background: var(--accent-surface);
  border-left-color: var(--accent);
  font-weight: 500;
}

.nav-icon {
  width: 16px;
  text-align: center;
  font-size: 0.8rem;
  opacity: 0.7;
}

.sidebar-footer {
  padding: 16px 20px;
  border-top: 1px solid var(--border-subtle);
  font-size: 0.75rem;
  color: var(--text-faint);
}

.sidebar-footer a { color: var(--text-muted); }
.sidebar-footer a:hover { color: var(--accent); }

/* Main content */
.main {
  margin-left: var(--sidebar-w);
  flex: 1;
  min-width: 0;
}

.content {
  max-width: 780px;
  margin: 0 auto;
  padding: 48px 40px 80px;
}

/* Typography */
h1 {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.2;
  margin-bottom: 8px;
  background: linear-gradient(135deg, var(--text) 0%, var(--text-muted) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

h2 {
  font-size: 1.45rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-top: 48px;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border-subtle);
  color: var(--text);
}

h3 {
  font-size: 1.1rem;
  font-weight: 600;
  margin-top: 32px;
  margin-bottom: 12px;
  color: var(--text);
}

p { margin-bottom: 16px; color: var(--text-muted); }

.lead {
  font-size: 1.1rem;
  color: var(--text-muted);
  margin-bottom: 32px;
  line-height: 1.7;
}

/* Code */
code {
  font-family: var(--font-mono);
  font-size: 0.85em;
  background: var(--bg-code);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: 2px 6px;
}

pre {
  background: var(--bg-code);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px 24px;
  overflow-x: auto;
  margin: 16px 0 24px;
  font-size: 0.82rem;
  line-height: 1.65;
}

pre code {
  background: none;
  border: none;
  padding: 0;
  font-size: inherit;
  color: var(--text);
}

/* Keyword highlighting via classes */
.kw { color: #c678dd; }
.fn { color: #61afef; }
.ty { color: #e5c07b; }
.st { color: #98c379; }
.cm { color: #5c6370; font-style: italic; }
.mc { color: #56b6c2; }
.nr { color: #d19a66; }

/* Cards */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
  margin: 24px 0;
}

.card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 24px;
  transition: all 0.2s;
}

.card:hover {
  border-color: var(--accent-muted);
  background: var(--bg-surface-hover);
  transform: translateY(-1px);
}

.card-title {
  font-family: var(--font-mono);
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--accent);
  margin-bottom: 8px;
}

.card-desc {
  font-size: 0.85rem;
  color: var(--text-muted);
  line-height: 1.6;
}

.card-tag {
  display: inline-block;
  font-size: 0.65rem;
  font-weight: 600;
  color: var(--text-faint);
  background: var(--bg);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  padding: 1px 6px;
  margin-top: 12px;
}

/* Hero */
.hero {
  padding: 24px 0 40px;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 40px;
}

.hero h1 {
  font-size: 2.6rem;
  margin-bottom: 12px;
}

.hero-actions {
  display: flex;
  gap: 12px;
  margin-top: 24px;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 10px 20px;
  font-size: 0.85rem;
  font-weight: 600;
  border-radius: 8px;
  border: 1px solid var(--border);
  transition: all 0.15s;
  cursor: pointer;
  font-family: var(--font-sans);
}

.btn-primary {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}

.btn-primary:hover {
  background: #ea580c;
  color: #fff;
}

.btn-ghost {
  background: transparent;
  color: var(--text-muted);
}

.btn-ghost:hover {
  background: var(--bg-surface);
  color: var(--text);
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0 24px;
  font-size: 0.87rem;
}

th {
  text-align: left;
  padding: 10px 16px;
  font-weight: 600;
  color: var(--text-faint);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  border-bottom: 1px solid var(--border);
}

td {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border-subtle);
  color: var(--text-muted);
}

tr:hover td { background: var(--bg-surface); }

/* Status badges */
.status {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 4px;
  letter-spacing: 0.02em;
}

.status-done { background: rgba(34, 197, 94, 0.12); color: #4ade80; }
.status-wip { background: rgba(249, 115, 22, 0.12); color: var(--accent); }
.status-planned { background: rgba(139, 139, 150, 0.1); color: var(--text-faint); }

/* Callout */
.callout {
  display: flex;
  gap: 12px;
  background: var(--accent-surface);
  border: 1px solid rgba(249, 115, 22, 0.2);
  border-radius: 8px;
  padding: 16px 20px;
  margin: 24px 0;
  font-size: 0.87rem;
  color: var(--text-muted);
}

.callout-icon { font-size: 1.1rem; flex-shrink: 0; }

/* Architecture diagram */
.arch-diagram {
  background: var(--bg-code);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 24px;
  margin: 24px 0;
  font-family: var(--font-mono);
  font-size: 0.78rem;
  line-height: 1.6;
  overflow-x: auto;
  white-space: pre;
  color: var(--text-muted);
}

.arch-diagram .highlight { color: var(--accent); font-weight: 600; }

/* Responsive */
@media (max-width: 860px) {
  .sidebar { display: none; }
  .main { margin-left: 0; }
  .content { padding: 32px 20px 60px; }
  .hero h1 { font-size: 2rem; }
  .card-grid { grid-template-columns: 1fr; }
}
`;
