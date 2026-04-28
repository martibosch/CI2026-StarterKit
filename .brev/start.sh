#!/bin/bash

# Bootstrap script for the CI 2026 Hackathon Brev launchable.
# Paste the raw GitHub URL of this file into the Brev launchable's
# startup script field. It clones (or updates) the repo and then
# delegates to the full setup.sh inside it, so pushing to the repo
# is all that's needed to update setup on new instances.
set -euo pipefail

REPO_URL="https://github.com/martibosch/CI2026-StarterKit.git"
REPO_DIR="/home/ubuntu/workspace"

if [ -d "$REPO_DIR/.git" ]; then
    echo "[start] Repo already cloned, pulling latest..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[start] Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo "[start] Delegating to setup.sh..."
bash "$REPO_DIR/.brev/setup.sh"
