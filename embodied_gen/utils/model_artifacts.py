# Project 3DGENERATION
# Modifications Copyright (c) 2025-2026 3DGENERATION Contributors.
# Licensed under the Apache License, Version 2.0. See ../../LICENSE.

"""Download locations for public model artifacts used by the WebUI."""

from __future__ import annotations

import os
from pathlib import Path

from torch.hub import download_url_to_file


SAM_CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/"
    f"{SAM_CHECKPOINT_NAME}"
)
AESTHETIC_CHECKPOINT_NAME = "sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_CHECKPOINT_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    f"raw/refs/heads/main/{AESTHETIC_CHECKPOINT_NAME}"
)


def model_assets_root() -> Path:
    """Return the project-local model artifact directory."""
    configured = os.getenv("EMBODIEDGEN_MODEL_ASSETS_DIR")
    if configured:
        return Path(configured).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "models" / "checkpoints"


def ensure_public_artifact(
    *, url: str, destination: Path, offline: bool
) -> Path:
    """Return an existing artifact or download it atomically from upstream."""
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    if offline:
        raise FileNotFoundError(
            f"Required model artifact is missing: {destination}. "
            "Run `python scripts/prefetch_models.py` before starting in "
            "offline mode."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        download_url_to_file(url, str(temporary), progress=True)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def ensure_sam_checkpoint(*, offline: bool) -> Path:
    """Resolve the official Segment Anything ViT-H checkpoint."""
    return ensure_public_artifact(
        url=SAM_CHECKPOINT_URL,
        destination=model_assets_root() / "sam" / SAM_CHECKPOINT_NAME,
        offline=offline,
    )


def ensure_aesthetic_checkpoint(*, offline: bool) -> Path:
    """Resolve the Improved Aesthetic Predictor checkpoint."""
    return ensure_public_artifact(
        url=AESTHETIC_CHECKPOINT_URL,
        destination=(
            model_assets_root()
            / "aesthetic"
            / AESTHETIC_CHECKPOINT_NAME
        ),
        offline=offline,
    )
