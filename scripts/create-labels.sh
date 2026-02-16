#!/usr/bin/env bash
# Create recommended gh labels for stevedores-org/llama.rs.
# Run from repo root with: ./scripts/create-labels.sh
# Requires: gh auth login (or GH_TOKEN)

set -e
REPO="${REPO:-stevedores-org/llama.rs}"

create() {
  local name="$1" color="$2" desc="$3"
  if gh label view "$name" --repo "$REPO" &>/dev/null; then
    echo "Label '$name' exists, skipping."
  else
    gh label create "$name" --repo "$REPO" --color "$color" --description "$desc"
    echo "Created label: $name"
  fi
}

create "cursor"       "0E8A16" "Work from Cursor IDE / AI-assisted"
create "llama.rs"      "FEF2C0" "In scope for stevedores-org/llama.rs"
create "docs"          "0052CC" "Documentation, roadmap, architecture"
create "roadmap"       "5319E7" "Roadmap / planning / priorities"
create "priority"      "B60205" "Priority / P1 items"
create "rag"           "1D76DB" "RAG / semantic search / oxidizedRAG"
create "good first issue" "7057FF" "Good for new contributors"

echo "Done. Use: gh pr create --base main --head cursor/featA --label cursor --label llama.rs --label docs"
echo "Or: gh issue list --label cursor --label llama.rs"
