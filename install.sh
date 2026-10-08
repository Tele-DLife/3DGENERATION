#!/bin/bash
# Based on EmbodiedGen, Copyright (c) Horizon Robotics and its contributors.
# Modifications Copyright (c) 2025-2026 3DGENERATION Contributors.
# Licensed under the Apache License, Version 2.0. See LICENSE.

set -e
# Full environment bootstrap (conda + submodules + this script): bash scripts/bootstrap_env.sh

STAGE=${1:-basic}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$SCRIPT_DIR/install/_utils.sh"
git config --global http.postBuffer 524288000

log_info "===== Starting installation stage: $STAGE ====="

if [[ "$STAGE" != "basic" ]]; then
    log_error "Unsupported installation stage: $STAGE (expected: basic)"
    exit 1
fi

bash "$SCRIPT_DIR/install/install_basic.sh"

pip install triton==3.5.1 numpy==1.26.4

log_info "===== Installation completed successfully. ====="
