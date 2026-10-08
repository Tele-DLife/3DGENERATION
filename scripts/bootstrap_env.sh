#!/usr/bin/env bash
# Based on EmbodiedGen, Copyright (c) Horizon Robotics and its contributors.
# Modifications Copyright (c) 2025-2026 3DGENERATION Contributors.
# Licensed under the Apache License, Version 2.0. See ../LICENSE.

# One-shot: create/update conda env, init submodules, run install.sh
#
# Usage:
#   bash scripts/bootstrap_env.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="${1:-basic}"
if [[ "$STAGE" != "basic" ]]; then
  echo "Usage: $0 [basic]" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Install Miniconda/Mambaforge, then re-run:" >&2
  echo "  https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME="3dgeneration"
if conda run -n "$ENV_NAME" true 2>/dev/null; then
  echo "[INFO] Updating conda env: $ENV_NAME"
  conda env update -f environment.yml --prune -n "$ENV_NAME" -y
else
  echo "[INFO] Creating conda env: $ENV_NAME"
  conda env create -f environment.yml -y
fi

conda activate "$ENV_NAME"

# Reduce ~/.local site-packages bleeding into this env (matches apps that prefer clean envs)
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

echo "[INFO] Initializing git submodules..."
git submodule update --init --recursive --progress

echo "[INFO] Running install.sh $STAGE (this may take a long time)..."
bash install.sh "$STAGE"

echo "[INFO] Done. Activate later with: conda activate $ENV_NAME"
