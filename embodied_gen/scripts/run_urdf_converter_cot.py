"""Run the CoT URDF pipeline (ranked-friction prompt + optional VLM height refine) end-to-end.

This script uses the real configured LLM/VLM client (``GPT_CLIENT``) and real mesh rendering
(``render_asset3d`` + optional ``pyrender`` for the reference-cube snapshot). It does not
start Gradio or ``apps/app_demo_CHN.py``.

Before importing Torch-dependent modules, this script:

- Sets ``CUDA_HOME`` / ``PATH`` / ``LD_LIBRARY_PATH`` so JIT builds find ``cuda_runtime_api.h``.
- Points ``CC``/``CXX`` at ``/usr/bin/gcc-12`` (or ``gcc-11``) when present, because CUDA 12.x
  ``nvcc`` rejects conda's GCC (>12) as ``-ccbin`` host compiler (see ``crt/host_config.h``).

Opt-out of compiler override: ``export EMBODIEDGEN_KEEP_CONDA_COMPILER=1``.

If a previous JIT build failed mid-way, remove the stale build dir under
``~/.cache/torch_extensions`` for ``nvdiffrast*`` once, then rerun.

Configure credentials the same way as the rest of the repo (``embodied_gen/utils/gpt_config.yaml``
and/or ``ENDPOINT`` / ``API_KEY`` / ``MODEL_NAME`` environment variables).

Example (from repository root):

    python -m embodied_gen.scripts.run_urdf_converter_cot \\
        --mesh path/to/sample.obj \\
        --output-root /tmp/urdf_cot_out \\
        --category marker \\
        --text-prompt ""
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _ensure_cuda_home_for_torch_extensions() -> None:
    """Local to this script only: help cpp_extension find CUDA headers (no repo-wide hooks)."""
    cur = (os.environ.get("CUDA_HOME") or "").strip()
    if cur and os.path.isfile(os.path.join(cur, "include", "cuda_runtime_api.h")):
        return
    if cur and os.path.isfile(
        os.path.join(
            cur, "targets", "x86_64-linux", "include", "cuda_runtime_api.h"
        )
    ):
        return

    candidates = (
        "/usr/local/cuda",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.2",
        "/usr/local/cuda-12",
        "/usr/local/cuda-11.8",
        "/opt/cuda",
    )
    root = None
    for c in candidates:
        if os.path.isfile(os.path.join(c, "include", "cuda_runtime_api.h")):
            root = c
            break
        alt = os.path.join(
            c, "targets", "x86_64-linux", "include", "cuda_runtime_api.h"
        )
        if os.path.isfile(alt):
            root = c
            break
    if root is None:
        nvcc = shutil.which("nvcc")
        if nvcc:
            cand = os.path.dirname(os.path.dirname(os.path.abspath(nvcc)))
            if os.path.isfile(os.path.join(cand, "include", "cuda_runtime_api.h")):
                root = cand
    if not root:
        return

    os.environ["CUDA_HOME"] = root
    bin_dir = os.path.join(root, "bin")
    if os.path.isdir(bin_dir):
        path = os.environ.get("PATH", "")
        parts = path.split(os.pathsep) if path else []
        if bin_dir not in parts:
            os.environ["PATH"] = bin_dir + (os.pathsep + path if path else "")
    lib64 = os.path.join(root, "lib64")
    if os.path.isdir(lib64):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        parts = ld.split(os.pathsep) if ld else []
        if lib64 not in parts:
            os.environ["LD_LIBRARY_PATH"] = lib64 + (os.pathsep + ld if ld else "")


def _ensure_nvcc_host_compiler_for_extensions() -> None:
    """nvcc uses env CC as -ccbin; conda gcc 13+ breaks CUDA 12.2 host_config check."""
    if os.environ.get("EMBODIEDGEN_KEEP_CONDA_COMPILER"):
        return
    cc_cur = (os.environ.get("CC") or "").strip()
    if cc_cur and "conda" not in cc_cur.lower() and "miniconda" not in cc_cur.lower():
        return
    pairs = [
        ("/usr/bin/gcc-12", "/usr/bin/g++-12"),
        ("/usr/bin/gcc-11", "/usr/bin/g++-11"),
    ]
    for cc, cxx in pairs:
        if os.path.isfile(cc) and os.path.isfile(cxx):
            os.environ["CC"] = cc
            os.environ["CXX"] = cxx
            os.environ.setdefault("CUDAHOSTCXX", cxx)
            os.environ.setdefault("CMAKE_CUDA_HOST_COMPILER", cxx)
            # Prefer system tools early so ``which gcc``-style probes match nvcc expectations.
            path = os.environ.get("PATH", "")
            prefix = "/usr/bin:/usr/local/bin"
            if not path.startswith(prefix):
                os.environ["PATH"] = prefix + (os.pathsep + path if path else "")
            return


_ensure_cuda_home_for_torch_extensions()
_ensure_nvcc_host_compiler_for_extensions()

from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.tags import VERSION
from embodied_gen.validators.urdf_converter_CoT import URDFGeneratorRankedFriction
from embodied_gen.validators.urdf_convertor import URDFGenerator


def _parse_height_range(s: str | None) -> dict:
    if not (s or "").strip():
        return {}
    parts = s.strip().split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid --height-range (expected min-max): {s!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if hi < lo:
        raise ValueError("height-range: max must be >= min")
    return {"min_height": lo, "max_height": hi}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real end-to-end run of URDFGeneratorRankedFriction (no Gradio)."
    )
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to input mesh (e.g. sample.obj from your pipeline).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Directory to write URDF, mesh copy, urdf_renders, height_sanity_check, etc.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="unknown",
        help="Object category label (forced into URDF when not 'unknown').",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="",
        help="Optional SAM3 / disambiguation hint appended to the main prompt.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=VERSION,
        help="Asset version string written into extra_info.",
    )
    parser.add_argument(
        "--decompose-convex",
        action="store_true",
        help="Use convex collision decomposition (same option as UI pipeline).",
    )
    parser.add_argument(
        "--no-height-vlm",
        action="store_true",
        help="Disable the second VLM height sanity step (only first LLM + URDF generation).",
    )
    parser.add_argument(
        "--reference-cube-m",
        type=float,
        default=0.10,
        help="Reference cube edge length in meters for the sanity snapshot (default 0.10).",
    )
    parser.add_argument(
        "--height-range",
        type=str,
        default="",
        help="Optional 'min-max' in meters to override model height after first LLM parse "
        "(same meaning as UI height range; empty = use model output only).",
    )
    args = parser.parse_args()

    mesh = Path(args.mesh).resolve()
    if not mesh.is_file():
        print(f"error: mesh file not found: {mesh}", file=sys.stderr)
        return 2

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    extra = _parse_height_range(args.height_range or None)

    gen = URDFGeneratorRankedFriction(
        GPT_CLIENT,
        render_view_num=4,
        decompose_convex=bool(args.decompose_convex),
        enable_height_vlm_refine=not args.no_height_vlm,
        reference_cube_size_m=float(args.reference_cube_m),
    )

    print("Calling URDFGeneratorRankedFriction (real GPT_CLIENT + real renders)...")
    urdf_path = gen(
        mesh_path=str(mesh),
        output_root=str(out_root),
        text_prompt=(args.text_prompt or "").strip(),
        category=(args.category or "unknown").strip(),
        version=args.version,
        **extra,
    )

    print(f"URDF written: {urdf_path}")
    if getattr(gen, "estimated_attrs", None) is not None:
        print(f"estimated_attrs: {gen.estimated_attrs}")

    snap = out_root / "height_sanity_check" / "mesh_with_reference_cube.png"
    if snap.is_file():
        print(f"height sanity snapshot: {snap}")
    elif not args.no_height_vlm:
        print(
            "note: height sanity snapshot missing (pyrender/EGL may have failed; "
            "heights were not VLM-refined)."
        )

    rh = URDFGenerator.get_attr_from_urdf(urdf_path, attr_name="real_height")
    print(f"extra_info real_height: {rh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
