#!/usr/bin/env python3
# Based on EmbodiedGen, Copyright (c) Horizon Robotics and its contributors.
# Modifications Copyright (c) 2025-2026 3DGENERATION Contributors.
# Licensed under the Apache License, Version 2.0. See ../LICENSE.

"""
Prefetch common, non-gated model artifacts needed by the WebUI.

Goal: avoid runtime downloads (which may fail on flaky networks) by caching
the reusable public artifacts ahead of first use. Gated SAM 3D Objects
checkpoints must be downloaded separately after accepting their license terms.

Typical usage (Image->3D app):
  python scripts/prefetch_models.py

By default, caches are placed under:
  ./models/cache/{torch,hf,u2net}
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _set_default_caches(root: Path) -> dict[str, str]:
    """
    Configure cache env vars so downloads go under the project directory.
    """
    cache_root = root / "models" / "cache"
    torch_home = cache_root / "torch"
    hf_home = cache_root / "hf"
    u2net_home = cache_root / "u2net"
    _mkdir(torch_home)
    _mkdir(hf_home)
    _mkdir(u2net_home)

    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
    # rembg/u2net cache
    os.environ.setdefault("U2NET_HOME", str(u2net_home))

    return {
        "TORCH_HOME": os.environ["TORCH_HOME"],
        "HF_HOME": os.environ["HF_HOME"],
        "HF_HUB_CACHE": os.environ["HF_HUB_CACHE"],
        "HUGGINGFACE_HUB_CACHE": os.environ["HUGGINGFACE_HUB_CACHE"],
        "U2NET_HOME": os.environ["U2NET_HOME"],
    }


def _retry(fn, *, tries: int, sleep_s: float, name: str) -> None:
    last_err: Exception | None = None
    for i in range(1, tries + 1):
        try:
            fn()
            return
        except Exception as e:
            last_err = e
            if i == tries:
                break
            _log(f"[prefetch] {name} failed (attempt {i}/{tries}): {e}")
            time.sleep(sleep_s * i)
    raise RuntimeError(f"{name} failed after {tries} attempts: {last_err}")


def prefetch_trellis(*, root: Path, trellis_local: Path, trellis_repo: str, tries: int) -> None:
    if trellis_local.exists() and (trellis_local / "pipeline.json").exists():
        _log(f"[prefetch] TRELLIS already present: {trellis_local}")
        return

    _mkdir(trellis_local)

    def _do():
        from huggingface_hub import snapshot_download

        _log(f"[prefetch] Downloading TRELLIS to {trellis_local} from {trellis_repo!r} ...")
        snapshot_download(
            repo_id=trellis_repo,
            local_dir=str(trellis_local),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    _retry(_do, tries=tries, sleep_s=3.0, name="TRELLIS snapshot_download")


def prefetch_moge(*, moge_dir: Path, moge_repo: str, tries: int) -> None:
    model_file = moge_dir / "model.pt"
    if model_file.exists() and model_file.stat().st_size > 0:
        _log(f"[prefetch] MoGe already present: {model_file}")
        return

    _mkdir(moge_dir)

    def _do():
        from huggingface_hub import snapshot_download

        _log(f"[prefetch] Downloading MoGe to {moge_dir} from {moge_repo!r} ...")
        snapshot_download(
            repo_id=moge_repo,
            local_dir=str(moge_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        if not model_file.exists():
            raise FileNotFoundError(f"Expected MoGe checkpoint not found: {model_file}")

    _retry(_do, tries=tries, sleep_s=3.0, name="MoGe snapshot_download")


def prefetch_dinov2(*, model_name: str, tries: int) -> None:
    # Avoid redundant downloads if the weights are already present in either
    # default torch hub cache or the configured TORCH_HOME. If present in the
    # default cache, copy them into TORCH_HOME to keep everything local.
    torch_home = Path(os.environ.get("TORCH_HOME", "")).expanduser()
    if str(torch_home) in ("", "."):
        torch_home = Path.home() / ".cache" / "torch"

    # For dinov2_vitl14_reg, torch.hub downloads this checkpoint name:
    expected_ckpt = torch_home / "hub" / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    expected_repo = torch_home / "hub" / "facebookresearch_dinov2_main"
    if expected_ckpt.exists() and expected_repo.exists():
        _log(f"[prefetch] DINOv2 already present: {expected_ckpt}")
        return

    # If TORCH_HOME is set to a project-local cache, but the user already has
    # DINOv2 in the global torch cache, copy it over to avoid downloading.
    default_torch_home = Path.home() / ".cache" / "torch"
    if default_torch_home != torch_home:
        src_ckpt = (
            default_torch_home
            / "hub"
            / "checkpoints"
            / "dinov2_vitl14_reg4_pretrain.pth"
        )
        src_repo = default_torch_home / "hub" / "facebookresearch_dinov2_main"
        if src_ckpt.exists() and src_repo.exists():
            _mkdir(expected_ckpt.parent)
            _mkdir(expected_repo.parent)
            shutil.copy2(src_ckpt, expected_ckpt)
            if expected_repo.exists():
                shutil.rmtree(expected_repo)
            shutil.copytree(src_repo, expected_repo)
            _log(
                f"[prefetch] Copied existing DINOv2 torch.hub cache into TORCH_HOME: {expected_ckpt}"
            )
            return

    def _do():
        import torch

        _log(f"[prefetch] torch.hub.load(dinov2, {model_name!r}, pretrained=True) ...")
        m = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        m.eval()

    _retry(_do, tries=tries, sleep_s=3.0, name=f"DINOv2 ({model_name})")


def prefetch_rembg_u2net(*, tries: int) -> None:
    # If user already has u2net cached in the default location, reuse it to
    # avoid a redundant download.
    u2net_home = Path(os.environ.get("U2NET_HOME", "")).expanduser()
    if str(u2net_home) == "." or str(u2net_home) == "":
        u2net_home = Path.home() / ".u2net"
    expected = u2net_home / "u2net.onnx"
    if expected.exists() and expected.stat().st_size > 0:
        _log(f"[prefetch] rembg u2net already present: {expected}")
        return

    # Try copying from common locations (rembg default is ~/.u2net).
    candidates = [
        Path.home() / ".u2net" / "u2net.onnx",
        Path.home() / ".cache" / "rembg" / "u2net.onnx",
    ]
    for src in candidates:
        if src.exists() and src.stat().st_size > 0:
            _mkdir(expected.parent)
            shutil.copy2(src, expected)
            _log(f"[prefetch] Copied existing u2net.onnx from {src} -> {expected}")
            return

    def _do():
        import rembg

        _log("[prefetch] rembg.new_session('u2net') ...")
        rembg.new_session("u2net")

    _retry(_do, tries=tries, sleep_s=2.0, name="rembg u2net")


def _download_url(*, url: str, destination: Path) -> None:
    from torch.hub import download_url_to_file

    _mkdir(destination.parent)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        download_url_to_file(url, str(temporary), progress=True)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def prefetch_sam(*, sam_dst: Path, url: str, tries: int) -> None:
    if sam_dst.exists():
        _log(f"[prefetch] SAM checkpoint already present: {sam_dst}")
        return

    _mkdir(sam_dst.parent)

    def _do():
        _log(f"[prefetch] Downloading SAM checkpoint from {url!r} ...")
        _download_url(url=url, destination=sam_dst)

    _retry(_do, tries=tries, sleep_s=3.0, name="SAM checkpoint download")


def prefetch_aesthetic(*, aesthetic_dir: Path, url: str, tries: int) -> None:
    sac = aesthetic_dir / "sac+logos+ava1-l14-linearMSE.pth"
    if sac.exists() and sac.stat().st_size > 0:
        _log(f"[prefetch] Aesthetic checkpoint already present: {sac}")
    else:
        _mkdir(aesthetic_dir)

        def _do_upstream():
            _log(
                f"[prefetch] Downloading aesthetic checkpoint from {url!r} ..."
            )
            _download_url(url=url, destination=sac)

        _retry(
            _do_upstream,
            tries=tries,
            sleep_s=3.0,
            name="Aesthetic checkpoint download",
        )

    def _do_clip():
        import clip

        _log("[prefetch] Ensuring CLIP weights are present ...")
        # AestheticPredictor expects CLIP weights in this directory; `clip.load`
        # will download missing files into download_root.
        clip.load("ViT-L/14", download_root=str(aesthetic_dir), device="cpu")

    _retry(_do_clip, tries=tries, sleep_s=3.0, name="CLIP weights")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check what is present/missing locally; do not download anything.",
    )
    parser.add_argument(
        "--tries",
        type=int,
        default=3,
        help="Retry count for flaky downloads.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="",
        help=(
            "Optional Hugging Face endpoint/mirror (sets HF_ENDPOINT). "
            "Example: https://hf-mirror.com (third-party mirror)."
        ),
    )
    parser.add_argument(
        "--skip-aesthetic",
        action="store_true",
        help="Skip aesthetic predictor assets (HF + CLIP weights).",
    )
    parser.add_argument(
        "--skip-dinov2",
        action="store_true",
        help="Skip DINOv2 torch.hub download.",
    )
    parser.add_argument(
        "--skip-rembg",
        action="store_true",
        help="Skip rembg u2net download.",
    )
    parser.add_argument(
        "--skip-moge",
        action="store_true",
        help="Skip MoGe weights used by SAM 3D Objects.",
    )
    parser.add_argument(
        "--trellis-repo",
        default="microsoft/TRELLIS-image-large",
        help="HF repo id for TRELLIS if local model folder is missing.",
    )
    parser.add_argument(
        "--moge-repo",
        default="Ruicheng/moge-vitl",
        help="HF repo id for the MoGe weights used by SAM 3D Objects.",
    )
    args = parser.parse_args()

    root = _repo_root()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
        _log(f"[prefetch] Using HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    caches = _set_default_caches(root)
    _log("[prefetch] Using caches:")
    for k, v in caches.items():
        _log(f"  - {k}={v}")

    trellis_local = root / "models" / "AI-ModelScope" / "TRELLIS-image-large"
    moge_dir = root / "weights" / "moge-vitl"
    model_assets_dir = Path(
        os.environ.get(
            "EMBODIEDGEN_MODEL_ASSETS_DIR", root / "models" / "checkpoints"
        )
    ).expanduser()
    sam_dst = model_assets_dir / "sam" / "sam_vit_h_4b8939.pth"
    aesthetic_dir = model_assets_dir / "aesthetic"

    if args.check:
        missing: list[str] = []
        trellis_ok = trellis_local.exists() and (trellis_local / "pipeline.json").exists()
        _log(f"[check] TRELLIS local: {'OK' if trellis_ok else 'MISSING'} ({trellis_local})")
        if not trellis_ok:
            missing.append("TRELLIS (models/AI-ModelScope/TRELLIS-image-large)")

        moge_file = moge_dir / "model.pt"
        moge_ok = moge_file.exists() and moge_file.stat().st_size > 0
        _log(f"[check] MoGe ViT-L: {'OK' if moge_ok else 'MISSING'} ({moge_file})")
        if not moge_ok and not args.skip_moge:
            missing.append("MoGe ViT-L (weights/moge-vitl/model.pt)")

        # DINOv2 torch.hub
        torch_home = Path(os.environ.get("TORCH_HOME", "")).expanduser()
        if str(torch_home) in ("", "."):
            torch_home = Path.home() / ".cache" / "torch"
        ckpt = torch_home / "hub" / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
        repo_dir = torch_home / "hub" / "facebookresearch_dinov2_main"
        dinov2_ok = ckpt.exists() and repo_dir.exists()
        _log(f"[check] DINOv2 hub+weights: {'OK' if dinov2_ok else 'MISSING'} ({ckpt})")
        if not dinov2_ok and not args.skip_dinov2:
            missing.append("DINOv2 (torch.hub: dinov2_vitl14_reg)")

        # rembg u2net
        u2net_home = Path(os.environ.get("U2NET_HOME", "")).expanduser()
        if str(u2net_home) in ("", "."):
            u2net_home = Path.home() / ".u2net"
        u2net = u2net_home / "u2net.onnx"
        u2net_ok = u2net.exists()
        _log(f"[check] rembg u2net: {'OK' if u2net_ok else 'MISSING'} ({u2net})")
        if not u2net_ok and not args.skip_rembg:
            missing.append("rembg u2net.onnx")

        # SAM checkpoint
        sam_ok = sam_dst.exists()
        _log(f"[check] SAM vit_h checkpoint: {'OK' if sam_ok else 'MISSING'} ({sam_dst})")
        if not sam_ok:
            missing.append("SAM checkpoint (sam_vit_h_4b8939.pth)")

        # Aesthetic predictor assets (optional)
        sac = aesthetic_dir / "sac+logos+ava1-l14-linearMSE.pth"
        aest_ok = sac.exists()
        _log(f"[check] Aesthetic predictor: {'OK' if aest_ok else 'MISSING'} ({sac})")
        if not aest_ok and not args.skip_aesthetic:
            missing.append("Aesthetic predictor assets (+ CLIP weights)")

        if missing:
            _log("[check] Missing artifacts:")
            for m in missing:
                _log(f"  - {m}")
            return 2
        _log("[check] All required artifacts are present.")
        return 0

    # Prefer local TRELLIS model folder; otherwise download.
    prefetch_trellis(
        root=root,
        trellis_local=trellis_local,
        trellis_repo=args.trellis_repo,
        tries=args.tries,
    )

    if not args.skip_moge:
        prefetch_moge(moge_dir=moge_dir, moge_repo=args.moge_repo, tries=args.tries)

    if not args.skip_dinov2:
        # The bundled TRELLIS pipeline.json specifies this:
        prefetch_dinov2(model_name="dinov2_vitl14_reg", tries=args.tries)

    if not args.skip_rembg:
        prefetch_rembg_u2net(tries=args.tries)

    # SAM is used by the point-click segmentation tool.
    prefetch_sam(
        url=(
            "https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_h_4b8939.pth"
        ),
        sam_dst=sam_dst,
        tries=args.tries,
    )

    if not args.skip_aesthetic:
        prefetch_aesthetic(
            url=(
                "https://github.com/christophschuhmann/"
                "improved-aesthetic-predictor/raw/refs/heads/main/"
                "sac+logos+ava1-l14-linearMSE.pth"
            ),
            aesthetic_dir=aesthetic_dir,
            tries=args.tries,
        )

    _log("[prefetch] Done.")
    _log("[prefetch] Next: run the app normally, it should not need network for these assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
