#!/bin/bash

# Brev setup script for the CI 2026 Hackathon Starter Kit.
# Runs automatically on first boot before participants access the instance.
# Requires NVIDIA driver >= 570 (ships with Brev CUDA 12.8 base images).
set -euo pipefail

# ── 0. System dependencies ────────────────────────────────────────────────────
echo "[setup] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq curl unzip

REPO_DIR="/home/ubuntu/workspace"

# ── 1. Pixi ───────────────────────────────────────────────────────────────────
if ! command -v pixi &>/dev/null; then
    echo "[setup] Installing Pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi
PIXI_BIN=$(command -v pixi)

# ── 2. Pixi CUDA environment ──────────────────────────────────────────────────
echo "[setup] Installing Pixi CUDA environment..."
"$PIXI_BIN" install --manifest-path "$REPO_DIR"

# ── 3. Training data from HuggingFace ─────────────────────────────────────────
echo "[setup] Downloading training data with Snakemake..."
cd "$REPO_DIR"
"$PIXI_BIN" run --manifest-path "$REPO_DIR" snakemake --cores 1 download_data

echo "[setup] Done. Enter the environment with: pixi shell"
