#!/usr/bin/env bash
# setup_env.sh — Set up development environment
# Usage: bash scripts/setup_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Creating Python venv ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Installing base requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Copying .env.example ==="
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[INFO] Created .env from .env.example — fill in your credentials"
else
    echo "[SKIP] .env already exists"
fi

echo ""
echo "=== Done ==="
echo "Activate the venv with: source .venv/bin/activate"
echo "Download model assets with: bash scripts/download_assets.sh"
