#!/usr/bin/env bash
# Create the project venv (.venv) with uv. Home is shared with bwrcr740-4,
# so one venv works on both the login node and the GPU node.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv sync
