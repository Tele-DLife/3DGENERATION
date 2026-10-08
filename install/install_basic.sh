#!/bin/bash
# Based on EmbodiedGen, Copyright (c) Horizon Robotics and its contributors.
# Modifications Copyright (c) 2025-2026 3DGENERATION Contributors.
# Licensed under the Apache License, Version 2.0. See ../LICENSE.

set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_utils.sh"

PIP_INSTALL_PACKAGES=(
    "pip==24.0"
    "torch==2.4.0+cu118 torchaudio==2.4.0+cu118 torchvision==0.19.0+cu118 --index-url https://download.pytorch.org/whl/cu118"
    "xformers==0.0.27.post2+cu118 --index-url https://download.pytorch.org/whl/cu118"
    "-r requirements.txt --use-deprecated=legacy-resolver"
    # "flash-attn==2.7.0.post2"
    "utils3d@git+https://github.com/EasternJournalist/utils3d.git@9a4eb15"
    "clip@git+https://github.com/openai/CLIP.git"
    "segment-anything@git+https://github.com/facebookresearch/segment-anything.git@dca509f"
    "nvdiffrast@git+https://github.com/NVlabs/nvdiffrast.git@729261d"
    "kaolin@git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.16.0"
    "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    "MoGe@git+https://github.com/microsoft/MoGe.git@a8c3734"
)

for pkg in "${PIP_INSTALL_PACKAGES[@]}"; do
    try_install "Installing $pkg..." \
        "pip install $pkg" \
        "$pkg installation failed."
done

try_install "Installing EmbodiedGen..." \
    "pip install -e .[dev]" \
    "EmbodiedGen installation failed."
