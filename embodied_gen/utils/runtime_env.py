"""
Runtime environment guards.

EmbodiedGen is often run in managed Python environments (conda/venv), but many
machines also have user-site packages (~/.local/...) and system ROS Python paths
on `sys.path`. Mixing those with the active env can cause hard-to-debug runtime
errors (e.g. torch/torchvision ABI mismatch, spconv JIT builds).

These helpers make CLI entrypoints more robust by:
- pruning user-site paths from sys.path (opt-out via env var)
- setting env vars to reduce/disable JIT compilation for optional CUDA deps
"""

from __future__ import annotations

import os
import sys
from typing import Iterable


def _is_user_site_path(p: str, user_site: str | None) -> bool:
    if not p:
        return False
    if "/.local/" in p:
        return True
    if user_site and os.path.abspath(p) == os.path.abspath(user_site):
        return True
    return False


def prune_sys_path(*, keep: Iterable[str] = ("",)) -> None:
    """
    Remove common user-site entries from sys.path to avoid mixing environments.

    Opt-out by setting:
      EMBODIEDGEN_ALLOW_USER_SITE=1
    """
    if os.environ.get("EMBODIEDGEN_ALLOW_USER_SITE"):
        return

    try:
        import site

        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    keep_set = set(keep)
    new_path: list[str] = []
    for p in sys.path:
        if p in keep_set:
            new_path.append(p)
            continue
        if _is_user_site_path(p, user_site):
            continue
        new_path.append(p)
    sys.path = new_path


def set_default_env_vars() -> None:
    """
    Set conservative defaults to avoid expensive/fragile JIT compilation during import.
    """
    # spconv (used by SAM3D) may try to JIT-compile CUDA extensions at import time.
    os.environ.setdefault("SPCONV_DISABLE_JIT", "1")
    # cumm/pccm are used underneath spconv.
    os.environ.setdefault("CUMM_DISABLE_JIT", "1")

    # Limit parallel compilation for torch extensions.
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", os.environ["MAX_JOBS"])

    # Pin torch extension cache dir.
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", os.path.expanduser("~/.cache/torch_extensions")
    )

    # Help torch/cpp_extension discover the CUDA toolkit headers/libs when building
    # optional CUDA extensions (gsplat/nvdiffrast/etc). If CUDA_HOME isn't set,
    # builds can fail with missing cuda headers like cuda_runtime_api.h.
    if "CUDA_HOME" not in os.environ:
        candidates = (
            "/usr/local/cuda",
            "/usr/local/cuda-12.2",
            "/usr/local/cuda-12",
            "/usr/local/cuda-11.8",
            "/opt/cuda",
        )
        for c in candidates:
            if os.path.exists(os.path.join(c, "include", "cuda_runtime_api.h")):
                os.environ["CUDA_HOME"] = c
                break
            if os.path.exists(
                os.path.join(
                    c,
                    "targets",
                    "x86_64-linux",
                    "include",
                    "cuda_runtime_api.h",
                )
            ):
                os.environ["CUDA_HOME"] = c
                break

    # nvcc from CUDA 12.2 rejects very new host compilers (e.g. conda gcc 14),
    # which breaks JIT builds for CUDA extensions (gsplat/nvdiffrast/etc).
    # Prefer a system GCC that is known to be supported by the installed CUDA.
    if not os.environ.get("EMBODIEDGEN_KEEP_CONDA_COMPILER"):
        cc = os.environ.get("CC", "")
        cxx = os.environ.get("CXX", "")
        looks_like_conda_cc = (
            "x86_64-conda-linux-gnu-cc" in cc or "/conda" in cc or "/miniconda" in cc
        )
        looks_like_conda_cxx = (
            "x86_64-conda-linux-gnu-c++" in cxx or "/conda" in cxx or "/miniconda" in cxx
        )
        # Choose a system compiler pair supported by nvcc.
        sys_cc = None
        sys_cxx = None
        if os.path.exists("/usr/bin/gcc-12") and os.path.exists("/usr/bin/g++-12"):
            sys_cc, sys_cxx = "/usr/bin/gcc-12", "/usr/bin/g++-12"
        elif os.path.exists("/usr/bin/gcc-11") and os.path.exists("/usr/bin/g++-11"):
            sys_cc, sys_cxx = "/usr/bin/gcc-11", "/usr/bin/g++-11"

        if sys_cc and sys_cxx:
            if looks_like_conda_cc:
                os.environ["CC"] = sys_cc
            if looks_like_conda_cxx:
                os.environ["CXX"] = sys_cxx
            os.environ.setdefault("CUDAHOSTCXX", sys_cxx)


def early_runtime_setup() -> None:
    """
    Call this at the top of CLI entrypoints (before importing torch/torchvision/spconv).
    """
    set_default_env_vars()
    prune_sys_path()


