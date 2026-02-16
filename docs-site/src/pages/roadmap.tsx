import type { FC } from "hono/jsx";
import { Layout } from "../components/Layout";

export const RoadmapPage: FC = () => (
  <Layout title="Roadmap" activePath="/roadmap">
    <h1>Roadmap</h1>
    <p class="lead">
      Five milestones from "Hello Inference" to RAG + Agents. Twelve user
      stories across five epics, tracked on GitHub.
    </p>

    <h2>Milestones</h2>
    <table>
      <thead>
        <tr>
          <th>Milestone</th>
          <th>Goal</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>A &mdash; Hello Inference</strong></td>
          <td>Tiny model forward pass on CPU, greedy sampling, CLI generate</td>
          <td><span class="status status-wip">In Progress</span></td>
        </tr>
        <tr>
          <td><strong>B &mdash; KV Cache Correctness</strong></td>
          <td>Prefill + decode KV equivalence, streaming TokenStream</td>
          <td><span class="status status-wip">In Progress</span></td>
        </tr>
        <tr>
          <td><strong>C &mdash; Real Weight Loading</strong></td>
          <td>Safetensors integration, tensor name mapping, memory checks</td>
          <td><span class="status status-planned">Planned</span></td>
        </tr>
        <tr>
          <td><strong>D &mdash; Metal Enablement</strong></td>
          <td>Op parity suite, Metal backend behind parity gate</td>
          <td><span class="status status-planned">Planned</span></td>
        </tr>
        <tr>
          <td><strong>E &mdash; RAG + Agents</strong></td>
          <td>oxidizedRAG adapter, oxidizedgraph agent nodes</td>
          <td><span class="status status-planned">Planned</span></td>
        </tr>
      </tbody>
    </table>

    <h2>Epic 1: Repository Foundation</h2>
    <p>
      Establish the Rust workspace, define core traits, and implement
      deterministic tokenization.
    </p>
    <table>
      <thead>
        <tr>
          <th>Story</th>
          <th>Issue</th>
          <th>Description</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>LLAMA-001</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/4">#4</a></td>
          <td>Modular workspace and narrow-waist <code>LlamaEngine</code> trait</td>
          <td><span class="status status-done">Done</span></td>
        </tr>
        <tr>
          <td>LLAMA-002</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/5">#5</a></td>
          <td>Robust tokenizer with streaming UTF-8 support</td>
          <td><span class="status status-wip">WIP</span></td>
        </tr>
        <tr>
          <td>LLAMA-003</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/6">#6</a></td>
          <td>Foundational sampling (greedy, top-k/p, temperature)</td>
          <td><span class="status status-done">Done</span></td>
        </tr>
      </tbody>
    </table>

    <h2>Epic 2: Inference &amp; KV Cache</h2>
    <p>
      Implement graph-friendly model architectures and first-class memory
      management for token sequences.
    </p>
    <table>
      <thead>
        <tr>
          <th>Story</th>
          <th>Issue</th>
          <th>Description</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>LLAMA-004</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/7">#7</a></td>
          <td>First-class KV Cache (prefill, decode, paging)</td>
          <td><span class="status status-done">Done</span></td>
        </tr>
        <tr>
          <td>LLAMA-005</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/8">#8</a></td>
          <td>Llama 3 and Qwen model blocks in Rust</td>
          <td><span class="status status-wip">WIP</span></td>
        </tr>
        <tr>
          <td>LLAMA-006</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/9">#9</a></td>
          <td>Prefill vs. Decode verification suite</td>
          <td><span class="status status-done">Done</span></td>
        </tr>
      </tbody>
    </table>

    <h2>Epic 3: Hardware Acceleration</h2>
    <p>
      Integrate oxidizedMLX and enforce strict parity between CPU and Metal
      backends.
    </p>
    <table>
      <thead>
        <tr>
          <th>Story</th>
          <th>Issue</th>
          <th>Description</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>LLAMA-007</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/10">#10</a></td>
          <td>Runtime selector for oxidizedMLX backends</td>
          <td><span class="status status-done">Done</span></td>
        </tr>
        <tr>
          <td>LLAMA-008</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/11">#11</a></td>
          <td>Backend Parity Gate (Metal vs CPU golden tests)</td>
          <td><span class="status status-planned">Planned</span></td>
        </tr>
      </tbody>
    </table>

    <h2>Epic 4: Application Surface</h2>
    <p>
      Expose the engine via a high-performance HTTP server with streaming,
      sessions, and cancellation.
    </p>
    <table>
      <thead>
        <tr>
          <th>Story</th>
          <th>Issue</th>
          <th>Description</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>LLAMA-009</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/12">#12</a></td>
          <td>OpenAI-compatible HTTP server</td>
          <td><span class="status status-planned">Planned</span></td>
        </tr>
        <tr>
          <td>LLAMA-010</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/13">#13</a></td>
          <td>Session and cancellation management</td>
          <td><span class="status status-planned">Planned</span></td>
        </tr>
      </tbody>
    </table>

    <h2>Epic 5: Agentic Compute</h2>
    <p>
      Integrate with the broader "oxidized" ecosystem for RAG and multi-agent
      orchestration.
    </p>
    <table>
      <thead>
        <tr>
          <th>Story</th>
          <th>Issue</th>
          <th>Description</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>LLAMA-011</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/14">#14</a></td>
          <td>RAG adapter for oxidizedRAG</td>
          <td><span class="status status-planned">Planned</span></td>
        </tr>
        <tr>
          <td>LLAMA-012</td>
          <td><a href="https://github.com/stevedores-org/llama.rs/issues/15">#15</a></td>
          <td>Agent orchestration via oxidizedgraph</td>
          <td><span class="status status-planned">Planned</span></td>
        </tr>
      </tbody>
    </table>
  </Layout>
);
