# Based on EmbodiedGen, Copyright (c) Horizon Robotics and its contributors.
# Modifications Copyright (c) 2025-2026 3DGENERATION Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import random
import signal
import select
import subprocess
import shutil
import base64
import time
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import tempfile
from pathlib import Path

from typing import Tuple, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROJECTS_DIR = _REPO_ROOT.parent

_ROBOTWIN_DIR = _PROJECTS_DIR / "RoboCousin"
_COUSIN_LAYOUT_SCRIPT = _ROBOTWIN_DIR / "cousin_layout" / "scripts" / "image_to_relative_layout_with_matching.py"
_COUSIN_LAYOUT_DESK_LAYOUT_ROOT = _ROBOTWIN_DIR / "cousin_layout" / "desk_layout"


# --- 1. 系统环境与底层工具函数 (保持原样) ---

def _find_conda_env_python(env_name: str) -> str | None:
    possible_conda_bases = [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        "/opt/conda",
        "/usr/local/miniconda3",
        "/usr/local/anaconda3",
    ]
    for conda_base in possible_conda_bases:
        candidate = os.path.join(conda_base, "envs", env_name, "bin", "python")
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None

def _minimal_subprocess_env(extra_keep: list[str] | None = None, extra_set: dict[str, str] | None = None) -> dict:
    keep = {
        "HOME", "USER", "LOGNAME", "SHELL", "PATH", "LANG", "LC_ALL", "TERM", "TZ",
        "DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
        "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "U2NET_HOME",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_EXPERIMENTAL_WARNING",
        "LOCAL_MODELS_ONLY", "EMBODIEDGEN_LOCAL_MODELS_ONLY", "ROBOTWIN_HF_CACHE_DIR",
    }
    if extra_keep: keep.update(extra_keep)
    env = {k: v for k, v in os.environ.items() if k in keep}
    hf_root = env.get("ROBOTWIN_HF_CACHE_DIR", os.environ.get("ROBOTWIN_HF_CACHE_DIR", "")).strip()
    if hf_root:
        hf_hub = os.path.join(hf_root, "hub")
        env["HF_HOME"] = hf_root
        env["HF_HUB_CACHE"] = hf_hub
        env["HUGGINGFACE_HUB_CACHE"] = hf_hub
        env["TRANSFORMERS_CACHE"] = hf_hub
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra_set: env.update(extra_set)
    return env


def _ensure_conda_libstdcpp():
    conda_prefix = os.getenv("CONDA_PREFIX")
    if not conda_prefix: return
    conda_lib = os.path.join(conda_prefix, "lib")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_path}" if ld_path else conda_lib
        os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_conda_libstdcpp()

def _disable_user_site():
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

_disable_user_site()

_cache_root = _REPO_ROOT / "models" / "cache"
_hf_home = str(_cache_root / "hf")
os.environ.setdefault("HF_HOME", _hf_home)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_hf_home, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_hf_home, "hub"))
os.environ.setdefault("TORCH_HOME", str(_cache_root / "torch"))
os.environ.setdefault("U2NET_HOME", str(_cache_root / "u2net"))
_local_models_only = os.getenv("LOCAL_MODELS_ONLY", os.getenv("EMBODIEDGEN_LOCAL_MODELS_ONLY", "1")) != "0"
os.environ["LOCAL_MODELS_ONLY"] = "1" if _local_models_only else "0"
os.environ["EMBODIEDGEN_LOCAL_MODELS_ONLY"] = os.environ["LOCAL_MODELS_ONLY"]

if _local_models_only or os.getenv("EMBODIEDGEN_OFFLINE", "1") != "0":
    os.environ["EMBODIEDGEN_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"


def _prefer_cuda_supported_host_compiler():
    gcc11 = "/usr/bin/gcc-11"
    gpp11 = "/usr/bin/g++-11"
    if os.path.exists(gcc11) and os.path.exists(gpp11):
        os.environ["CC"] = gcc11
        os.environ["CXX"] = gpp11
        os.environ["CUDAHOSTCXX"] = gpp11
        os.environ["CMAKE_CUDA_HOST_COMPILER"] = gpp11
        if "CUDACXX" not in os.environ:
            os.environ["CUDACXX"] = os.environ.get("CUDA_HOME", "/usr/local/cuda") + "/bin/nvcc"
        compiler_name = "g++-11"
        base_cache = os.path.expanduser("~/.cache/torch_extensions")
        os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(base_cache, compiler_name)
        os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
        for path_dir in os.environ.get("PATH", "").split(":"):
            conda_gcc = os.path.join(path_dir, "x86_64-conda-linux-gnu-gcc")
            if os.path.exists(conda_gcc):
                os.environ["PATH"] = "/usr/bin:/usr/local/bin:" + os.environ.get("PATH", "")
                break


_prefer_cuda_supported_host_compiler()
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
os.environ.setdefault("MAX_JOBS", "2")
os.environ.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
os.environ["GRADIO_APP"] = "imageto3d_sam3d"

# --- 2. 导入 Gradio 及其它依赖 ---

import gradio as gr
from app_style import custom_theme, image_css, lighting_css
from common import (
    MAX_SEED, VERSION, active_btn_by_content, end_session,
    extract_3d_representations_v3, extract_urdf,
    image_to_3d, select_point, start_session, SAM_PREDICTOR,
    TMP_DIR,
)

sample_step = 25
_active_robotwin_processes = {}

SAM3_REPO = _PROJECTS_DIR / "sam3"
SAM3_CLI = Path(__file__).parent / "sam3_segment_cli.py"
HF_CACHE_DIR = "/data/huggingface_cache/hub"
os.environ.setdefault("ROBOTWIN_HF_CACHE_DIR", str(Path(HF_CACHE_DIR).parent))

# 流式日志仍传完整缓冲（有上限），界面保持 lines=6；在文本框内滚轮可看更早行
_LOG_UI_MAX_LINES = 2000


def _format_log_lines(lines: list[str], prefix: str) -> str:
    buf = lines[-_LOG_UI_MAX_LINES:] if len(lines) > _LOG_UI_MAX_LINES else lines
    return "\n".join(f"{prefix} {l}" for l in buf)


def segment_with_sam3(
    image: Image.Image,
    prompt: str,
    confidence: float,
) -> tuple[Image.Image | None, Image.Image | None]:
    """Segment image using SAM3 via CLI. Returns (seg_rgba, overlay_rgb)."""
    if image is None or not prompt:
        return None, None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        image.save(tmp_in.name)
        tmp_in_path = tmp_in.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask:
        mask_path = tmp_mask.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_overlay:
        overlay_path = tmp_overlay.name

    try:
        conda_env = "sam3"
        cmd = [
            "conda",
            "run",
            "-n",
            conda_env,
            "python",
            str(SAM3_CLI),
            "--image",
            tmp_in_path,
            "--prompt",
            prompt,
            "--confidence",
            str(confidence),
            "--out-mask",
            mask_path,
            "--out-overlay",
            overlay_path,
            "--sam3-repo",
            str(SAM3_REPO),
            "--hf-cache-dir",
            HF_CACHE_DIR,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=_minimal_subprocess_env(),
        )
        if result.returncode != 0:
            print(f"SAM3 segmentation failed: {result.stderr}")
            return None, None

        mask_image = Image.open(mask_path).convert("L")
        overlay_image = Image.open(overlay_path).convert("RGB")
        rgba_image = image.convert("RGBA")
        rgba_image.putalpha(mask_image)
        return rgba_image, overlay_image
    except Exception as e:
        print(f"Error in SAM3 segmentation: {e}")
        return None, None
    finally:
        for p in (tmp_in_path, mask_path, overlay_path):
            try:
                os.unlink(p)
            except Exception:
                pass


def _terminate_robotwin_process_group(process, wait_timeout=8.0):
    if process is None or process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.terminate()
        except Exception:
            pass
    try:
        process.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except Exception:
                pass
        try:
            process.wait(timeout=5)
        except Exception:
            pass


# --- 3. 业务逻辑分离 ---

# A. 数字表亲专用逻辑
def _save_dt_image_for_digital_cousins(dt_img: np.ndarray, request: gr.Request) -> str:
    sh = str(request.session_hash)
    out_dir = os.path.join(TMP_DIR, sh)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dc_input.png")
    img = Image.fromarray(dt_img) if isinstance(dt_img, np.ndarray) else dt_img
    img.save(out_path)
    return out_path

def run_digital_cousins_layout_stream(input_image_path: str):
    """
    桌面数字表亲：在 robocousin conda env 中运行 vendored `cousin_layout` 脚本。

    说明：
    - 不再依赖 `acdc` env / `digital_cousins` 包入口（避免 IsaacSim bootstrap 等隐式差异）
    - 输出默认写到 `<RoboCousin>/cousin_layout/desk_layout/<timestamp>/...`
    """
    python = os.getenv("ROBOCOUSIN_PYTHON") or _find_conda_env_python("robocousin")
    if not python:
        yield "❌ 未找到 robocousin 解释器：请设置 ROBOCOUSIN_PYTHON 或创建 conda env「robocousin」"
        return
    if not _COUSIN_LAYOUT_SCRIPT.is_file():
        yield f"❌ 未找到 cousin_layout 启动脚本：{_COUSIN_LAYOUT_SCRIPT}"
        return

    save_dir = os.getenv("COUSIN_LAYOUT_SAVE_DIR") or str(_COUSIN_LAYOUT_DESK_LAYOUT_ROOT / str(time.time()))
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        python,
        str(_COUSIN_LAYOUT_SCRIPT),
        "--input-image-path",
        input_image_path,
        "--save-dir",
        save_dir,
        "--robo-twin-root",
        str(_ROBOTWIN_DIR),
        "--snapshot-overwrite",
        "--verbose",
        "--support-footprint-slice-frac",
        "0.2",
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
        cwd=str(_ROBOTWIN_DIR),
        env=_minimal_subprocess_env(),
        bufsize=1,
    )
    lines = []
    for line in iter(process.stdout.readline, ""):
        if line:
            lines.append(line.rstrip())
            yield _format_log_lines(lines, "🛠️")
    process.wait()

def _latest_dc_layout_json() -> str:
    root = os.getenv("COUSIN_LAYOUT_DESK_LAYOUT_ROOT") or str(_COUSIN_LAYOUT_DESK_LAYOUT_ROOT)
    if not os.path.exists(root): return ""
    subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subdirs: return ""
    latest = max(subdirs, key=lambda p: os.path.getmtime(p))
    p = os.path.join(latest, "relative_layout", "relative_layout_ontop_desktable.json")
    return p if os.path.exists(p) else ""

def _next_numeric_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    existing_ids = [
        int(d)
        for d in os.listdir(base_dir)
        if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))
    ]
    next_idx = max(existing_ids) + 1 if existing_ids else 0
    return os.path.join(base_dir, str(next_idx))


def _copy_urdf_sample_to(dest_dir: str, source_urdf: str, source_mesh_dir: str):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(source_urdf, os.path.join(dest_dir, "sample.urdf"))
    if os.path.exists(source_mesh_dir):
        shutil.copytree(source_mesh_dir, os.path.join(dest_dir, "mesh"))


# B. 3D 资产：URDF 同步（仅在 asset 模式按钮触发）
def copy_urdf_to_robotwin(request: gr.Request, target_kind: str):
    sh = str(request.session_hash)
    urdf_sample_dir = os.path.join(TMP_DIR, sh, "URDF_sample")
    source_urdf = os.path.join(urdf_sample_dir, "sample.urdf")
    source_mesh_dir = os.path.join(urdf_sample_dir, "mesh")
    ROBOTWIN_ASSETS_DIR = str(_PROJECTS_DIR / "RoboCousin" / "our_assets")
    ROBOTWIN_TEST_DIR = str(_PROJECTS_DIR / "RoboCousin" / "threeD-generation" / "obj")
    if not (os.path.exists(urdf_sample_dir) and os.path.exists(source_urdf)):
        print("❌ 未找到 URDF_sample，请检查 session 与 TMP_DIR。")
        return

    if target_kind not in ("actor", "non-actor", "room", "general"):
        print(f"❌ 未知 target_kind: {target_kind}")
        return

    category = "unknown"
    try:
        tree = ET.parse(source_urdf)
        cat_node = tree.getroot().find(".//extra_info/category")
        if cat_node is not None and cat_node.text:
            category = cat_node.text.strip().strip('"').strip("'")
    except Exception as e:
        print(f"⚠️ 提取 category 时出错: {e}")

    permanent_root = os.path.join(ROBOTWIN_ASSETS_DIR, target_kind, category)
    permanent_dir = _next_numeric_dir(permanent_root)

    # 1) 两种都同步到 test 目录
    _copy_urdf_sample_to(ROBOTWIN_TEST_DIR, source_urdf, source_mesh_dir)
    # 2) 再各自复制到 actor / non-actor 的永久目录
    _copy_urdf_sample_to(permanent_dir, source_urdf, source_mesh_dir)

    print("✅ RoboTwin 保存完成")
    print(f"- target_kind: {target_kind}")
    print(f"- category: {category}")
    print(f"- test_dir: {ROBOTWIN_TEST_DIR}")
    print(f"- permanent_dir: {permanent_dir}")


def full_pipeline_with_progress(
    raw_img, seg_img, seed, ss_sampling_steps, slat_sampling_steps,
    ss_guidance_strength, slat_guidance_strength, texture_size,
    asset_cat_text, height_range_text, seg_backend, sam3_prompt,
    request: gr.Request, progress=gr.Progress(),
):
    try:
        progress(0, desc="正在初始化模型...")
        if isinstance(raw_img, np.ndarray):
            raw_img = Image.fromarray(raw_img)
        elif not isinstance(raw_img, Image.Image):
            raw_img = Image.fromarray(np.array(raw_img))
        progress(0.1, desc="正在生成 3D 几何结构...")
        out_buf, video = image_to_3d(
            None, seed, ss_sampling_steps, slat_sampling_steps, raw_img,
            ss_guidance_strength, slat_guidance_strength, seg_img, True, req=request,
        )
        progress(0.7, desc="正在渲染 3D 表示与纹理...")
        mesh, gs, obj, aligned = extract_3d_representations_v3(
            out_buf, texture_size, req=request,
        )
        progress(0.9, desc="正在完成物理属性对齐...")
        # 规则：
        # - 填了「物体标签」：SAM / SAM3 均用标签作 VL hint，并强制 URDF category（与 extract_urdf 一致）
        # - 未填标签且 SAM3：仅用文字描述框作 hint
        # - SAM 点选模式：不使用文字框 prompt；有标签用标签，无标签则无 hint
        effective_cat = (asset_cat_text or "").strip()
        if effective_cat:
            hint_prompt = effective_cat
        elif seg_backend == "sam3":
            hint_prompt = (sam3_prompt or "").strip()
        else:
            hint_prompt = ""
        extract_urdf(
            aligned,
            obj,
            effective_cat,
            height_range_text or "",
            "",
            VERSION,
            text_prompt=hint_prompt,
            req=request,
        )
        progress(1.0, desc="生成完成！")
        return out_buf, video, mesh, gs, obj, aligned
    except Exception as e:
        import traceback
        print(f"Pipeline error: {e}")
        print(traceback.format_exc())
        if "out_buf" in locals() and "video" in locals():
            return out_buf, video, mesh, gs, obj, aligned
        raise

# 手动保存：URDF 生成完后由 Step3 按钮触发
def save_asset_to_robotwin(mode: str, target_kind: str, request: gr.Request):
    if mode != "asset_extraction":
        return gr.update(), gr.update(), gr.update(), gr.update()
    copy_urdf_to_robotwin(request, target_kind=target_kind)
    # 点击即保存；不在 UI 里额外输出日志
    # 任意一个被点击后，四个按钮都禁用（本轮只允许保存一次）
    disabled = gr.update(visible=True, interactive=False)
    return disabled, disabled, disabled, disabled


# C. 家庭场景搭建专用逻辑（独立于“3D 资产 / 桌面表亲”，只输出日志）
def run_scene_builder_pipeline(scene_type: str, template_path: str, request: gr.Request):
    scene_type = (scene_type or "").strip() or "客厅"
    template_path = (template_path or "").strip()

    python = os.getenv("ROBOCOUSIN_PYTHON") or _find_conda_env_python("robocousin")
    if not python:
        yield "❌ 未找到 robocousin 环境：请设置 ROBOCOUSIN_PYTHON 或创建 conda env「robocousin」"
        return

    robotwin_path = str(_PROJECTS_DIR / "RoboCousin")
    lines: list[str] = []

    def _emit(msg: str):
        lines.append(msg)
        return _format_log_lines(lines, "🏠")

    robotwin_layout_json = str(
        _PROJECTS_DIR / "RoboCousin" / "envs" / "room_config" / "ui_generated_layout.json"
    )
    os.makedirs(os.path.dirname(robotwin_layout_json), exist_ok=True)

    # 1) 布局/匹配（step3 只做到这里，渲染放到 step4）
    if template_path and os.path.isfile(template_path):
        yield _emit(f"📦 使用现有模板：{template_path}")
        try:
            shutil.copy2(template_path, robotwin_layout_json)
        except Exception as e:
            yield _emit(f"❌ 写入 RoboTwin 布局文件失败：{e}")
            return
        yield _emit(f"✅ 布局/匹配完成：{robotwin_layout_json}")
        return

    yield _emit(f"🏗️ 场景布局/匹配中：{scene_type}")
    match_cmd = [python, "-m", "envs.utils.layout_matcher", "--type", scene_type]
    process_gen = subprocess.Popen(
        match_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=robotwin_path,
        env=_minimal_subprocess_env(),
        bufsize=1,
    )
    for line in iter(process_gen.stdout.readline, ""):
        if line:
            yield _emit(f"📐 {line.rstrip()}")
    process_gen.wait()
    if process_gen.returncode != 0:
        yield _emit(f"❌ 布局生成失败（退出码 {process_gen.returncode}）")
        return
    if not os.path.isfile(robotwin_layout_json):
        # matcher 理应写入这个固定路径；如果没有写出来，给出明确提示
        yield _emit(f"⚠️ 未检测到 RoboTwin 布局文件：{robotwin_layout_json}")
        yield _emit("请检查 layout_matcher 的输出路径是否与 RoboTwin 渲染脚本一致。")
        return
    yield _emit(f"✅ 布局/匹配完成：{robotwin_layout_json}")

# D. 统一调度器：三路互不污染输出
def run_step2_pipeline(
    mode, r_img, s_img, d_img, s, ss, sl, ss_g, slat_g, tex, asset_cat_text, height_range_text, seg_backend, sam3_prompt,
    scene_type, scene_template, # 新增参数
    request: gr.Request, progress=gr.Progress(),
):
    noop_mesh, noop_video = gr.update(), gr.update()
    noop_dt, noop_scene = gr.update(), gr.update()
    empty_layout = ""
    if mode == "desktop_twin":
        path = _save_dt_image_for_digital_cousins(d_img, request)
        for msg in run_digital_cousins_layout_stream(path):
            yield msg, noop_mesh, noop_video, noop_scene, False, empty_layout
        return
    if mode == "scene_builder":
        for msg in run_scene_builder_pipeline(scene_type, scene_template, request):
            yield noop_dt, noop_mesh, noop_video, msg, False, empty_layout
        return
    yield noop_dt, noop_mesh, noop_video, "⏳ 3D 资产生成中…", False, empty_layout
    out_buf, video, mesh, gs, obj, aligned = full_pipeline_with_progress(
        r_img, s_img, s, ss, sl, ss_g, slat_g, tex,
        asset_cat_text, height_range_text, seg_backend, sam3_prompt,
        request, progress=progress,
    )
    yield noop_dt, mesh, video, noop_scene, True, empty_layout

def _step4_scene_to_robotwin_room_env(scene_preset: str | None) -> dict[str, str]:
    """
    Step4 场景名 → RoboTwin 子进程环境（envs.utils.ui_room_preset_env / preview_cousin_layout_ui）。
    极简风：ROBOTWIN_UI_ROOM_TYPE 置空（不加房间家具）；其余房间 → 对应 preset 目录 + 随机 JSON。
    """
    sp = (scene_preset or "").strip()
    out: dict[str, str] = {"ROBOTWIN_UI_ROOM_TYPE": ""}
    if sp and sp != "极简风":
        _map = {
            "客厅": "livingroom",
            "卧室": "bedroom",
            "餐厅": "diningroom",
            "卫生间": "bathroom",
            "儿童房": "kidsroom",
            "书房": "study",
        }
        rt = _map.get(sp, "")
        if rt:
            out["ROBOTWIN_UI_ROOM_TYPE"] = rt
    return out


# E. Step 4：资产模式用 robocousin Python 跑数采；数字表亲用同一解释器跑 layout 预览
# scene_preset: Step4 点选房间；极简风 = 不加 UI 房间背景（与原先 run_obj_data_collection 一致）
def run_robotwin_bash(request: gr.Request, scene_preset: str | None = None):
    path = str(_PROJECTS_DIR / "RoboCousin")
    py_entry = os.path.join(path, "threeD-generation", "run_obj_data_collection.py")
    sh_script = os.path.join(path, "threeD-generation", "run_obj_data_collection.sh")
    python = os.getenv("ROBOCOUSIN_PYTHON") or _find_conda_env_python("robocousin")
    room_env = _step4_scene_to_robotwin_room_env(scene_preset)
    room_type = room_env.pop("ROBOTWIN_UI_ROOM_TYPE", "")
    if python and os.path.isfile(py_entry):
        cmd = [python, py_entry]
    else:
        cmd = ["bash", sh_script]
    if room_type:
        cmd.extend(["--room-type", room_type])
    extra_env = {"PYTHONUNBUFFERED": "1", **room_env}
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=0,
        cwd=path, env=_minimal_subprocess_env(extra_set=extra_env), start_new_session=True,
    )
    sh = str(request.session_hash)
    _active_robotwin_processes[sh] = process
    lines = []
    try:
        while sh in _active_robotwin_processes:
            ready, _, _ = select.select([process.stdout], [], [], 0.5)
            if not ready:
                continue
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
            line = line.rstrip()
            if line:
                lines.append(line)
            yield _format_log_lines(lines, "📡")
    finally:
        _active_robotwin_processes.pop(sh, None)
    if process.returncode == 0:
        yield "✅ 数采任务结束"
    else:
        yield f"❌ 进程退出码 {process.returncode}"


def run_step4_dispatch(mode, layout_path, scene_preset: str | None, request: gr.Request):
    # 家庭场景搭建：不依赖 Step4 场景点选；渲染带导出（与 app_demo_UI_random 一致）
    if mode == "scene_builder":
        robotwin_dir = str(_PROJECTS_DIR / "RoboCousin")
        python = os.getenv("ROBOCOUSIN_PYTHON") or _find_conda_env_python("robocousin")
        if not python:
            yield "❌ 未找到 robocousin 环境：请设置 ROBOCOUSIN_PYTHON 或安装 conda env robocousin"
            return

        robotwin_layout_json = str(
            _PROJECTS_DIR / "RoboCousin" / "envs" / "room_config" / "ui_generated_layout.json"
        )
        if not os.path.isfile(robotwin_layout_json):
            yield "❌ 未找到布局文件，请先在 Step3 完成布局/匹配。"
            return

        env = _minimal_subprocess_env(
            extra_set={
                "PYTHONUNBUFFERED": "1",
                "ROBOTWIN_VIEWER_RES": os.getenv("ROBOTWIN_VIEWER_RES", "1280,720"),
                "ROBOTWIN_VIEWER_PLACEMENT": "1100,500",
                "ROBOTWIN_VIEWER_MINIMAL_UI": "1",
            }
        )
        cmd = [python, "threeD-generation/ui_generated_scene_render_ui.py"]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=0,
            cwd=robotwin_dir,
            env=env,
            start_new_session=True,
        )
        sh = str(request.session_hash)
        _active_robotwin_processes[sh] = process
        lines = []
        try:
            while sh in _active_robotwin_processes:
                ready, _, _ = select.select([process.stdout], [], [], 0.5)
                if not ready:
                    continue
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                line = line.rstrip()
                if line:
                    lines.append(line)
                yield _format_log_lines(lines, "🖼️")
        finally:
            _active_robotwin_processes.pop(sh, None)
        return

    if not (scene_preset or "").strip():
        yield "❌ 请先在 Step4 上方点选场景（极简风或任一房间）"
        return
    scene_line = f"🏠 已选场景：{scene_preset.strip()}"
    if mode == "desktop_twin":
        robotwin_dir = str(_PROJECTS_DIR / "RoboCousin")
        python = os.getenv("ROBOCOUSIN_PYTHON") or _find_conda_env_python("robocousin")
        if not python:
            yield "❌ 未找到 robocousin 环境：请设置 ROBOCOUSIN_PYTHON 或安装 conda env robocousin"
            return
        if not layout_path or not os.path.isfile(layout_path):
            yield "❌ 无效的 layout 路径，请先完成桌面数字表亲同步。"
            return
        yield scene_line
        cmd = [
            python, "threeD-generation/preview_cousin_layout_ui.py",
            "--input_layout", layout_path, "--task-config", "demo_complete", "--no-arm",
            "--seed", str(random.randrange(2**31)),
        ]
        # 与资产数采一致：极简风不加房间；否则由 preview 内读取 ROBOTWIN_UI_ROOM_TYPE 随机选布局 JSON
        env = _minimal_subprocess_env(
            extra_set={
                "PYTHONUNBUFFERED": "1",
                **_step4_scene_to_robotwin_room_env(scene_preset),
                "ROBOTWIN_VIEWER_RES": os.getenv("ROBOTWIN_VIEWER_RES", "1280,720"),
                "ROBOTWIN_VIEWER_PLACEMENT": "1100,500",
                "ROBOTWIN_VIEWER_MINIMAL_UI": "1",
            }
        )
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=0,
            cwd=robotwin_dir, env=env, start_new_session=True,
        )
        sh = str(request.session_hash)
        _active_robotwin_processes[sh] = process
        lines = []
        try:
            while sh in _active_robotwin_processes:
                ready, _, _ = select.select([process.stdout], [], [], 0.5)
                if not ready:
                    continue
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                line = line.rstrip()
                if line:
                    lines.append(line)
                yield _format_log_lines(lines, "📡")
        finally:
            _active_robotwin_processes.pop(sh, None)
        return
    yield scene_line
    yield from run_robotwin_bash(request, scene_preset)


def stop_robotwin_bash(request: gr.Request):
    sh = str(request.session_hash)
    if sh in _active_robotwin_processes:
        p = _active_robotwin_processes.pop(sh)
        _terminate_robotwin_process_group(p)
        return "⏹️ 已停止"
    return "ℹ️ 无活跃进程"

# --- 4. UI 静态组件 ---

LOGO_PATH = str(Path(__file__).resolve().parent.parent / "tianyi.png")
def _logo_html(p):
    if not os.path.exists(p): return ""
    with open(p, "rb") as f: enc = base64.b64encode(f.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{enc}" style="width:200px; display:block;">'

def _section_header(n, t):
    return f'''<div style="display:flex; align-items:center; margin-bottom:12px; border-bottom:1px solid #cbd5e1; padding-bottom:8px;"><div style="color:#1B4596; font-family:\'Courier New\',monospace; font-weight:900; margin-right:12px;">[ 0{n} ]</div><h3 style="margin:0; font-weight:800; color:#0f172a; font-size:1.1em;">{t}</h3></div>'''

structured_css = """
<style>
    body { background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important; padding: 20px 0 !important; }
    .gradio-container { background-color: #ffffff !important; max-width: 1440px !important; border-radius: 20px !important; padding: 30px !important; }
    .pro-card { background: #f8fafc !important; border-radius: 12px !important; border: 1px solid #e2e8f0 !important; border-top: 4px solid #1B4596 !important; padding: 16px !important; margin-bottom: 16px !important; }
    
    /* 强力锁定背景，杜绝初始化变黑 */
    .terminal-log textarea { background-color: #0f172a !important; color: #10b981 !important; font-family: monospace !important; border-radius: 6px !important; font-size: 12px !important; resize: none !important; overflow-y: auto !important; }
    .clean-log, .clean-log textarea, .clean-log .container { background-color: #ffffff !important; background: #ffffff !important; }
    .clean-log textarea { color: #1B4596 !important; font-family: monospace !important; border: 1px solid #cbd5e1 !important; border-radius: 6px !important; font-size: 12px !important; resize: none !important; overflow-y: auto !important; }
    
    #glow-btn { background: linear-gradient(135deg, #1B4596 0%, #2a61ce 100%) !important; color: white !important; font-weight: 800 !important; animation: glow 1.5s infinite alternate; }
    @keyframes glow { 0% { box-shadow: 0 0 5px rgba(27,69,150,0.2); } 100% { box-shadow: 0 0 15px rgba(27,69,150,0.5); } }

    /* Step4：场景点选行（在 Runtime Shell 之上） */
    #step4-room-row {
        margin-bottom: 10px !important;
        width: 88% !important;          /* 收窄整排按钮，避免“横向过长” */
        max-width: 980px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        overflow: hidden !important;
        display: grid !important;
        grid-template-columns: repeat(7, minmax(0, 1fr)) !important; /* 永远 7 列 */
        gap: 5px !important;
        align-items: stretch !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: 1px solid #d7e1f2 !important;
        background:
            linear-gradient(180deg, rgba(248,251,255,0.95) 0%, rgba(241,246,255,0.92) 100%),
            radial-gradient(120% 180% at 8% -20%, rgba(59,130,246,0.14) 0%, rgba(59,130,246,0) 55%) !important;
        box-shadow: 0 8px 22px rgba(15, 47, 102, 0.08), 0 1px 0 rgba(255,255,255,0.95) inset !important;
    }
    #step4-room-row > .column,
    #step4-room-row .column {
        min-width: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    /* 「建立仿真 / 阻断进程」与上方 Shell 的间距；整体上移可调小 clamp 的三个值 */
    .robotwin-actions-row {
        margin-top: clamp(0.75rem, 2.5vh, 1.5rem) !important;
        padding-top: 0.5rem !important;
        align-items: stretch !important;
    }
    .robotwin-actions-row > .column {
        flex: 1 1 0 !important;
        min-width: 0 !important;
    }
    .robotwin-actions-row.rail-btn-surface button {
        font-size: 13.5px !important;
        padding: 11px 16px !important;
        min-height: 44px !important;
    }

    /* 与 Step3 资产栏一致的按钮表面（套在 Row / Column 上） */
    .rail-btn-surface .wrap { gap: 6px !important; width: 100% !important; }
    .rail-btn-surface button {
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        font-weight: 650 !important;
        letter-spacing: 0.01em !important;
        padding: 8px 10px !important;
        line-height: 1.35 !important;
        transition: transform 0.15s ease, box-shadow 0.2s ease, border-color 0.15s ease !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 0 rgba(255,255,255,0.9) inset !important;
    }
    .rail-btn-surface button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(27, 69, 150, 0.12) !important;
        border-color: #94a3b8 !important;
    }
    .rail-btn-surface .primary button,
    .rail-btn-surface button.primary {
        background: linear-gradient(135deg, #1B4596 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border-color: #1e3a8a !important;
        font-weight: 700 !important;
    }
    .rail-btn-surface .primary button:hover,
    .rail-btn-surface button.primary:hover {
        box-shadow: 0 6px 18px rgba(27, 69, 150, 0.28) !important;
        border-color: #1B4596 !important;
    }
    .rail-btn-surface .stop button,
    .rail-btn-surface button.stop {
        background: #fff1f2 !important;
        color: #991b1b !important;
        border-color: #fecdd3 !important;
        font-weight: 700 !important;
    }
    .rail-btn-surface .stop button:hover,
    .rail-btn-surface button.stop:hover {
        box-shadow: 0 4px 14px rgba(185, 28, 28, 0.15) !important;
        border-color: #f87171 !important;
    }
    /* Gradio 部分版本将 stop 样式挂在列上，兜底与 Step3 白底描边体系一致 */
    .robotwin-actions-row.rail-btn-surface > .column:last-child button {
        background: #fff1f2 !important;
        color: #991b1b !important;
        border-color: #fecdd3 !important;
        font-weight: 700 !important;
    }
    .robotwin-actions-row.rail-btn-surface > .column:last-child button:hover {
        box-shadow: 0 4px 14px rgba(185, 28, 28, 0.15) !important;
        border-color: #f87171 !important;
    }
    #step4-room-row .wrap {
        gap: 0 !important;
    }
    #step4-room-row button {
        width: 100% !important;
        max-width: 100% !important;
        justify-content: center !important;
        text-align: center !important;
        font-size: clamp(11px, 0.82vw, 12.5px) !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
        padding: 6px 6px !important;
        min-height: 35px !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        border-radius: 10px !important;
        border: 1px solid #c6d4ea !important;
        color: #1a365d !important;
        background:
            linear-gradient(180deg, #ffffff 0%, #f3f7ff 100%),
            radial-gradient(130% 150% at 50% 0%, rgba(96,165,250,0.16) 0%, rgba(96,165,250,0) 72%) !important;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.95) inset,
            0 0 0 1px rgba(59,130,246,0.06),
            0 3px 10px rgba(30, 64, 175, 0.10) !important;
        transition: all 0.18s ease !important;
    }
    #step4-room-row button:hover {
        transform: translateY(-1px) !important;
        border-color: #7ca4dd !important;
        color: #112f52 !important;
        background:
            linear-gradient(180deg, #ffffff 0%, #eaf2ff 100%),
            radial-gradient(130% 150% at 50% 0%, rgba(59,130,246,0.2) 0%, rgba(59,130,246,0) 74%) !important;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.95) inset,
            0 0 0 1px rgba(59,130,246,0.18),
            0 8px 16px rgba(30, 64, 175, 0.16) !important;
    }
    #step4-room-row .primary button,
    #step4-room-row button.primary {
        background:
            linear-gradient(180deg, #2f72d9 0%, #2056aa 62%, #1a468e 100%),
            radial-gradient(120% 170% at 50% -15%, rgba(191,219,254,0.42) 0%, rgba(191,219,254,0) 64%) !important;
        color: #f8fafc !important;
        border-color: #17407f !important;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.2) inset,
            0 0 0 1px rgba(191,219,254,0.16),
            0 10px 20px rgba(23, 64, 127, 0.35) !important;
        font-weight: 700 !important;
    }
    #step4-room-row .primary button:hover,
    #step4-room-row button.primary:hover {
        border-color: #2b60b8 !important;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.18) inset,
            0 0 0 1px rgba(147,197,253,0.22),
            0 8px 18px rgba(15, 47, 102, 0.38) !important;
    }
    #step4-room-row button:disabled {
        border-color: #e2e8f0 !important;
        color: #a3afc2 !important;
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%) !important;
        box-shadow: 0 1px 0 rgba(255,255,255,0.9) inset, 0 1px 3px rgba(148,163,184,0.15) !important;
        cursor: not-allowed !important;
        opacity: 1 !important;
        transform: none !important;
    }

    /* 强制“原图/预览”并排，不允许换行 */
    .seg-pair-row { flex-wrap: nowrap !important; }
    .seg-pair-row > .column { min-width: 0 !important; }

    /* Step1（scene_builder）：与 Step4 统一的卡片与按钮语言 */
    .scene-type-dual-row {
        gap: 10px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .scene-type-card {
        border-radius: 12px !important;
        border: 1px solid #d7e1f2 !important;
        padding: 10px !important;
        background:
            linear-gradient(180deg, rgba(248,251,255,0.95) 0%, rgba(241,246,255,0.92) 100%),
            radial-gradient(120% 180% at 8% -20%, rgba(59,130,246,0.14) 0%, rgba(59,130,246,0) 55%) !important;
        box-shadow: 0 8px 22px rgba(15, 47, 102, 0.08), 0 1px 0 rgba(255,255,255,0.95) inset !important;
    }
    .scene-type-card-title {
        margin: 0 0 8px 2px !important;
        font-size: 12.5px !important;
        font-weight: 800 !important;
        color: #1a365d !important;
        letter-spacing: 0.03em !important;
    }
    .scene-type-btn-row {
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
        gap: 6px !important;
        margin-top: 8px !important;
    }
    .scene-type-btn-row button {
        min-height: 35px !important;
        border-radius: 10px !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
    }
    .scene-type-btn-row.scene-type-supported-row button {
        border: 1px solid #c6d4ea !important;
        color: #1a365d !important;
        background:
            linear-gradient(180deg, #ffffff 0%, #f3f7ff 100%),
            radial-gradient(130% 150% at 50% 0%, rgba(96,165,250,0.14) 0%, rgba(96,165,250,0) 72%) !important;
        box-shadow: 0 1px 0 rgba(255,255,255,0.95) inset, 0 0 0 1px rgba(59,130,246,0.05), 0 3px 10px rgba(30,64,175,0.08) !important;
        transition: all 0.18s ease !important;
    }
    .scene-type-btn-row.scene-type-supported-row button:hover {
        transform: translateY(-1px) !important;
        border-color: #7ca4dd !important;
        color: #112f52 !important;
        background:
            linear-gradient(180deg, #ffffff 0%, #eaf2ff 100%),
            radial-gradient(130% 150% at 50% 0%, rgba(59,130,246,0.2) 0%, rgba(59,130,246,0) 74%) !important;
        box-shadow: 0 1px 0 rgba(255,255,255,0.95) inset, 0 0 0 1px rgba(59,130,246,0.18), 0 8px 16px rgba(30,64,175,0.16) !important;
    }
    .scene-type-btn-row.scene-type-supported-row .primary button,
    .scene-type-btn-row.scene-type-supported-row button.primary {
        border-color: #17407f !important;
        color: #f8fafc !important;
        background:
            linear-gradient(180deg, #2f72d9 0%, #2056aa 62%, #1a468e 100%),
            radial-gradient(120% 170% at 50% -15%, rgba(191,219,254,0.42) 0%, rgba(191,219,254,0) 64%) !important;
        box-shadow: 0 1px 0 rgba(255,255,255,0.2) inset, 0 0 0 1px rgba(191,219,254,0.16), 0 10px 20px rgba(23,64,127,0.35) !important;
    }
    .scene-type-btn-row.scene-type-disabled-row button {
        border: 1px solid #e2e8f0 !important;
        color: #a3afc2 !important;
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%) !important;
        box-shadow: 0 1px 0 rgba(255,255,255,0.9) inset, 0 1px 3px rgba(148,163,184,0.12) !important;
        cursor: not-allowed !important;
        opacity: 1 !important;
    }

    /* Step3：3D 预览 + 右侧竖排资产分类操作条 */
    .asset-rail-heading {
        font-size: 12px !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        margin: 0 0 2px 1px !important;
        padding-bottom: 8px !important;
        border-bottom: 1px solid #e2e8f0 !important;
        letter-spacing: 0.06em !important;
        font-family: ui-sans-serif, system-ui, sans-serif !important;
    }
    .asset-step3-row {
        align-items: stretch !important;
        gap: 10px !important;
        flex-wrap: nowrap !important;
    }
    .asset-step3-row > .column:first-child { min-width: 0 !important; flex: 1 1 auto !important; }
    .asset-export-rail {
        flex: 0 0 auto !important;
        width: min(240px, 28vw) !important;
        max-width: 260px !important;
        box-sizing: border-box !important;
        padding: 10px 10px !important;
        border-radius: 10px !important;
        background: linear-gradient(165deg, #ffffff 0%, #f1f5f9 55%, #e8eef5 100%) !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.85), 0 1px 2px rgba(15, 23, 42, 0.06) !important;
        border-left: 3px solid #1B4596 !important;
    }
    .asset-export-rail.rail-btn-surface .wrap > * { width: 100% !important; }

</style>
"""

# --- 5. UI 构建 ---

with gr.Blocks(theme=custom_theme, title="具身智能仿真训练平台") as demo:
    gr.HTML(image_css + lighting_css + structured_css)
    current_mode = gr.State("asset_extraction")
    layout_path_state = gr.State("")

    # [Header]
    with gr.Row(elem_classes=["header-row"]):
        logo = _logo_html(LOGO_PATH)
        gr.HTML(f'''<div style="position:relative; text-align:center; padding-bottom:15px;"><div style="position:absolute; top:0; left:10px;">{logo}</div><div style="padding-top:55px;"><h1 style="margin:0; font-size:2.4em; font-weight:900; color:#1B4596; letter-spacing:4px;">具身智能泛家庭仿真训练平台</h1><p style="margin:8px 0 0 0; color:#475569; font-size:1.1em; font-weight:600; letter-spacing:2px; font-family:monospace;">[ EMBODIED AI SIMULATION PLATFORM ]</p></div></div>''')

    with gr.Row():
        # --- 左栏 ---
        with gr.Column(scale=4):
            with gr.Column(elem_classes=["pro-card"]):
                gr.HTML(_section_header("1", "任务模式与输入"))
                with gr.Tabs() as mode_tabs:
                    with gr.Tab("3D 资产生成", id="asset_extraction"):
                        with gr.Row(elem_classes=["seg-pair-row"]):
                            with gr.Column(scale=1):
                                img_sam = gr.Image(label="原图", type="numpy", height=220)
                            with gr.Column(scale=1):
                                msk_sam = gr.AnnotatedImage(label="预览", height=220, visible=True)
                                msk_sam3 = gr.Image(label="预览", type="pil", height=220, visible=False)

                        # 分割方式选择与交互控件（放在图片下方）
                        with gr.Tabs() as seg_tabs:
                            with gr.Tab("点选", id="sam"):
                                fg_bg = gr.Radio(
                                    ["foreground_point", "background_point"],
                                    label="点选模式",
                                    value="foreground_point",
                                )
                            with gr.Tab("文字描述", id="sam3"):
                                sam3_prompt = gr.Textbox(
                                    label="Prompt",
                                    placeholder="用一句话描述要分割的物体，例如：a chair / the sofa / the plant",
                                    value="",
                                )
                                sam3_conf = gr.Slider(
                                    0.0,
                                    1.0,
                                    label="置信度阈值",
                                    value=0.5,
                                    step=0.05,
                                )
                                sam3_segment_btn = gr.Button("🧩 生成分割预览", variant="secondary")
                                sam3_hint = gr.HTML(visible=False)
                        with gr.Accordion("⚙️ 参数配置", open=False):
                            with gr.Row():
                                asset_cat_text = gr.Textbox(label="物体标签")
                                height_range_text = gr.Textbox(label="高度 (m)（如 0.5-0.6）")
                            seed = gr.Slider(0, MAX_SEED, label="Seed", value=0)
                            tex_size = gr.Slider(1024, 4096, label="分辨率", value=2048)
                            ss_steps = gr.Slider(1, 50, label="形状步数", value=25)
                            slat_steps = gr.Slider(1, 50, label="外观步数", value=25)
                            ss_g = gr.Slider(0.0, 10.0, label="结构引导", value=7.5)
                            slat_g = gr.Slider(0.0, 10.0, label="外观引导", value=3.0)
                    with gr.Tab("桌面数字表亲", id="desktop_twin"):
                        img_dt = gr.Image(label="全景图", type="numpy", height=300)
                    
                    # 修改后：Radio 选项
                    with gr.Tab("家庭场景搭建", id="scene_builder"):
                        with gr.Group(elem_classes=["pro-card"]):
                            with gr.Column(elem_classes=["scene-type-dual-row"]):
                                with gr.Column(elem_classes=["scene-type-card"]):
                                    gr.HTML('<div class="scene-type-card-title">家庭场景</div>')
                                    with gr.Row(elem_classes=["scene-type-btn-row", "scene-type-supported-row"]):
                                        scene_type_living_btn = gr.Button("客厅", variant="primary")
                                        scene_type_bedroom_btn = gr.Button("卧室", variant="secondary")
                                        scene_type_dining_btn = gr.Button("餐厅", variant="secondary")
                                    with gr.Row(elem_classes=["scene-type-btn-row", "scene-type-supported-row"]):
                                        scene_type_bath_btn = gr.Button("卫生间", variant="secondary")
                                        scene_type_children_btn = gr.Button("儿童房", variant="secondary")
                                        scene_type_study_btn = gr.Button("书房", variant="secondary")
                                with gr.Column(elem_classes=["scene-type-card"]):
                                    gr.HTML('<div class="scene-type-card-title">泛家庭场景</div>')
                                    with gr.Row(elem_classes=["scene-type-btn-row", "scene-type-disabled-row"]):
                                        gr.Button("保安亭", variant="secondary", interactive=False)
                                        gr.Button("健身房", variant="secondary", interactive=False)
                                        gr.Button("社区休闲设施", variant="secondary", interactive=False)
                            # gr.Markdown("💡 选择场景后点击下方按钮开始布局匹配与渲染")
                        with gr.Accordion("🛠️ 现有场景配置", open=False):
                            scene_template = gr.Textbox(
                                label="现有场景模板路径",
                                placeholder="/path/to/your/room/config/living_room.json",
                                value="",
                            )

            with gr.Column(elem_classes=["pro-card"]):
                gr.HTML(_section_header("2", "引擎执行管线"))
                gen_btn = gr.Button("🚀 启动重构/同步", variant="primary", elem_id="glow-btn", interactive=False)

        # --- 右侧 ---
        with gr.Column(scale=7):
            with gr.Column(elem_classes=["pro-card"]):
                gr.HTML(_section_header("3", "重构状态检验"))
                with gr.Column(visible=True) as asset_view:
                    with gr.Row(elem_classes=["asset-step3-row"], equal_height=True):
                        with gr.Column(scale=5, min_width=0):
                            with gr.Tabs():
                                with gr.Tab("3D 拓扑模型"): out_mesh = gr.Model3D(height=200, elem_id="lighter_mesh")
                                with gr.Tab("过程回放"): out_video = gr.Video(height=200)
                        with gr.Column(scale=1, min_width=200, elem_classes=["asset-export-rail", "rail-btn-surface"]):
                            gr.HTML('<div class="asset-rail-heading">资产分类</div>')
                            asset_save_actor_btn = gr.Button(
                                "Actor · 可操作物体",
                                variant="primary",
                                visible=False,
                                interactive=False,
                            )
                            asset_save_nonactor_btn = gr.Button(
                                "Static · 背景物体",
                                variant="secondary",
                                visible=False,
                                interactive=False,
                            )
                            asset_save_room_btn = gr.Button(
                                "Room · 场景家具",
                                variant="secondary",
                                visible=False,
                                interactive=False,
                            )
                            asset_save_general_btn = gr.Button(
                                "General · 泛家庭家具",
                                variant="secondary",
                                visible=False,
                                interactive=False,
                            )
                with gr.Column(visible=False) as dt_view:
                    dt_sync_logs = gr.Textbox(label="桌面表亲排布同步日志", lines=10, max_lines=10, interactive=False, elem_classes=["clean-log"])
                    dt_success_msg = gr.HTML('<div style="color:#15803d; font-weight:900; text-align:center;">✅ 场景同步完毕</div>', visible=False)
                with gr.Column(visible=False) as scene_view:
                    scene_logs = gr.Textbox(
                        label="场景构建日志", 
                        lines=10, 
                        max_lines=10, 
                        interactive=False, 
                        elem_classes=["clean-log"]
                    )
                    scene_success_msg = gr.HTML(
                        '<div style="color:#15803d; font-weight:900; text-align:center;">✅ 布局/匹配完成</div>',
                        visible=False,
                    )
            with gr.Column(elem_classes=["pro-card"]):
                gr.HTML(_section_header("4", "下游仿真与通信"))
                with gr.Row(elem_id="step4-room-row", elem_classes=["step4-room-row", "rail-btn-surface"]):
                    step4_room_raw_btn = gr.Button("极简风", variant="secondary", visible=False)
                    step4_room_living_btn = gr.Button("客厅", variant="secondary", visible=False)
                    step4_room_bedroom_btn = gr.Button("卧室", variant="secondary", visible=False)
                    step4_room_dining_btn = gr.Button("餐厅", variant="secondary", visible=False)
                    step4_room_bath_btn = gr.Button("卫生间", variant="secondary", visible=False)
                    step4_room_kids_btn = gr.Button("儿童房", variant="secondary", visible=False)
                    step4_room_study_btn = gr.Button("书房", variant="secondary", visible=False)
                logs = gr.Textbox(label="Runtime Shell", interactive=False, visible=False, lines=6, max_lines=6, elem_classes=["terminal-log"])
                with gr.Row(elem_classes=["robotwin-actions-row", "rail-btn-surface"]):
                    dc_btn = gr.Button("▶️ 建立仿真", variant="secondary", visible=False)
                    stop_btn = gr.Button("⏹️ 阻断进程", variant="stop", visible=False)

    # --- 6. 核心逻辑绑定 ---
    raw_cache = gr.State()
    orig_img_np_state = gr.State()
    sam_img_np_state = gr.State()
    sam_scale_state = gr.State()
    selected_points = gr.State([])
    image_seg_sam = gr.Image(visible=False, image_mode="RGBA", type="pil")
    seg_backend = gr.State("sam")
    sam3_prompt_state = gr.State("")
    asset_urdf_ready_state = gr.State(False)
    step4_scene_preset = gr.State(None)
    scene_type_state = gr.State("客厅")

    def _step4_idle_for_twin_or_asset():
        """切到表亲/资产：Step4 回到「先点房间再出建立仿真」的初始态（避免从场景搭建带过去渲染按钮）。"""
        room_hide = (gr.update(visible=False),) * 7
        return room_hide + (
            gr.update(visible=False),  # logs
            gr.update(visible=False, value="▶️ 建立仿真"),  # dc_btn
            gr.update(visible=False),  # stop_btn
            None,  # step4_scene_preset
        )

    def _step4_idle_for_scene_builder():
        """切到场景搭建：隐藏房间键；渲染/阻断等 Step3 完成后再由 _gen_done_ui 打开。"""
        room_hide = (gr.update(visible=False),) * 7
        return room_hide + (
            gr.update(visible=False),  # logs
            gr.update(visible=False, value="▶️ 渲染场景"),  # dc_btn
            gr.update(visible=False),  # stop_btn
            None,
        )

    def on_mode_select(evt: gr.SelectData):
        if evt.value == "桌面数字表亲":
            return (
                "desktop_twin",
                gr.update(interactive=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            ) + _step4_idle_for_twin_or_asset()
        if evt.value in ("家庭场景搭建", "泛家庭场景搭建"):
            return (
                "scene_builder",
                gr.update(interactive=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            ) + _step4_idle_for_scene_builder()
        return (
            "asset_extraction",
            gr.update(),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        ) + _step4_idle_for_twin_or_asset()

    mode_tabs.select(
        on_mode_select,
        None,
        [
            current_mode,
            gen_btn,
            asset_view,
            dt_view,
            scene_view,
            step4_room_raw_btn,
            step4_room_living_btn,
            step4_room_bedroom_btn,
            step4_room_dining_btn,
            step4_room_bath_btn,
            step4_room_kids_btn,
            step4_room_study_btn,
            logs,
            dc_btn,
            stop_btn,
            step4_scene_preset,
        ],
    )

    def _set_seg_backend(evt: gr.SelectData, orig_np, sam_np):
        # evt.value 是 Tab 的显示标题（这里是“点选”/“文字描述”）
        v = str(evt.value)
        backend = "sam3" if ("文字" in v or "SAM3" in v) else "sam"

        # 切换 tab 不做 SAM 预处理（避免 SAM3 模式浪费时间）；SAM 的预处理推迟到第一次点选
        show_img = orig_np if orig_np is not None else gr.update()
        sam_np_out = sam_np

        # 切换分割方式后清理点/预览，避免混用旧状态
        return (
            backend,
            gr.update(visible=(backend == "sam")),
            gr.update(visible=(backend == "sam3")),
            show_img,
            sam_np_out,
            [],
            gr.update(value=None),
            gr.update(value=None),  # msk_sam
            gr.update(value=None),  # msk_sam3
        )

    seg_tabs.select(
        _set_seg_backend,
        inputs=[orig_img_np_state, sam_img_np_state],
        outputs=[
            seg_backend,
            msk_sam,   # visible toggle
            msk_sam3,  # visible toggle
            img_sam,
            sam_img_np_state,
            selected_points,
            image_seg_sam,
            msk_sam,   # clear value
            msk_sam3,  # clear value
        ],
    )
    sam3_prompt.change(lambda x: x or "", inputs=[sam3_prompt], outputs=[sam3_prompt_state])

    # 资产图上传：不做 SAM 预处理（SAM 预处理推迟到第一次点选）
    def _on_asset_image_upload(img_np):
        if img_np is None:
            return (
                gr.update(),
                None,
                None,
                None,
                [],
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
            )
        if isinstance(img_np, np.ndarray):
            img_pil = Image.fromarray(img_np).convert("RGB")
        elif isinstance(img_np, Image.Image):
            img_pil = img_np.convert("RGB")
            img_np = np.array(img_pil)
        else:
            img_pil = Image.fromarray(np.array(img_np)).convert("RGB")
            img_np = np.array(img_pil)

        sam_np = None
        show_np = img_np

        # 清理交互状态与预览
        return (
            show_np,               # img_sam（原图）
            img_pil,               # raw_cache (真正原始 RGB，用于后续贴图/回写)
            img_np,                # orig_img_np_state
            sam_np,                # sam_img_np_state
            None,                  # sam_scale_state
            [],                    # selected_points
            gr.update(value=None), # image_seg_sam
            gr.update(value=None), # msk_sam
            gr.update(value=None), # msk_sam3
        )

    img_sam.upload(
        _on_asset_image_upload,
        inputs=[img_sam],
        outputs=[img_sam, raw_cache, orig_img_np_state, sam_img_np_state, sam_scale_state, selected_points, image_seg_sam, msk_sam, msk_sam3],
    )

    def _select_point_guarded(img, orig_np, sam_np, sam_scale, pts, fg_bg_mode, backend, evt: gr.SelectData):
        if backend != "sam":
            return gr.update(), gr.update(), sam_np, sam_scale

        if orig_np is None:
            return gr.update(), gr.update(), sam_np, sam_scale

        if sam_np is None:
            sam_np = SAM_PREDICTOR.preprocess_image(orig_np)
            try:
                if hasattr(SAM_PREDICTOR.predictor, "reset_image"):
                    SAM_PREDICTOR.predictor.reset_image()
            except Exception:
                pass
            try:
                SAM_PREDICTOR.predictor.set_image(sam_np)
            except Exception:
                pass
            oh, ow = orig_np.shape[:2]
            sh, sw = sam_np.shape[:2]
            sam_scale = (sw / max(1, ow), sh / max(1, oh))

        sx, sy = sam_scale if sam_scale else (1.0, 1.0)
        x, y = evt.index
        x2 = int(round(x * sx))
        y2 = int(round(y * sy))
        from types import SimpleNamespace
        evt2 = SimpleNamespace(index=(x2, y2))

        mask_anno, seg_img = select_point(sam_np, pts, fg_bg_mode, evt2)
        return mask_anno, seg_img, sam_np, sam_scale

    img_sam.select(
        _select_point_guarded,
        [img_sam, orig_img_np_state, sam_img_np_state, sam_scale_state, selected_points, fg_bg, seg_backend],
        [msk_sam, image_seg_sam, sam_img_np_state, sam_scale_state],
    )

    # SAM3 Prompt 分割 -> 统一产出 image_seg_sam
    def _sam3_preview(img_np, prompt, conf):
        if img_np is None:
            return gr.update(), gr.update(), gr.update(visible=False)
        if not (prompt or "").strip():
            return gr.update(), gr.update(), gr.update(visible=False)
        img_pil = Image.fromarray(img_np) if isinstance(img_np, np.ndarray) else img_np
        seg_rgba, overlay = segment_with_sam3(img_pil, prompt, conf)
        if seg_rgba is None:
            return gr.update(), gr.update(), gr.update(visible=False)
        try:
            # alpha 全 0 => 空分割：提示用户并保持生成按钮禁用（image_seg_sam 置空）
            alpha = np.array(seg_rgba.split()[-1])
            if alpha.size == 0 or int(alpha.max()) == 0:
                hint = (
                    "<div style='margin-top:6px; padding:8px 10px; border-radius:8px; "
                    "border:1px solid #f59e0b; background:#fffbeb; color:#92400e; font-weight:700;'>"
                    "SAM3 分割结果为空：请尝试降低置信度阈值或调整 prompt。</div>"
                )
                return overlay, gr.update(value=None), gr.update(value=hint, visible=True)
        except Exception:
            pass
        return overlay, seg_rgba, gr.update(visible=False)

    sam3_segment_btn.click(
        _sam3_preview,
        inputs=[img_sam, sam3_prompt, sam3_conf],
        outputs=[msk_sam3, image_seg_sam, sam3_hint],
    )

    image_seg_sam.change(active_btn_by_content, [image_seg_sam], [gen_btn])
    img_dt.change(lambda x: gr.update(interactive=True) if x is not None else gr.update(interactive=False), [img_dt], [gen_btn])
    def _scene_type_updates(choice: str):
        def _one(chosen: str, label: str):
            return gr.update(variant="primary" if chosen == label else "secondary")
        return (
            _one(choice, "客厅"),
            _one(choice, "卧室"),
            _one(choice, "餐厅"),
            _one(choice, "卫生间"),
            _one(choice, "儿童房"),
            _one(choice, "书房"),
            choice,
        )

    _step4_hide_all = (gr.update(visible=False),) * 7 + (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        None,
    )

    def _gen_start_ui(mode):
        busy = gr.update(interactive=False, value="⏳ 处理中...")
        if mode == "desktop_twin":
            return (
                busy,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(),
                gr.update(visible=False, interactive=False),  # asset_save_actor_btn
                gr.update(visible=False, interactive=False),  # asset_save_nonactor_btn
                gr.update(visible=False, interactive=False),  # asset_save_room_btn
                gr.update(visible=False, interactive=False),  # asset_save_general_btn
                False,  # asset_urdf_ready_state
            ) + _step4_hide_all
        if mode == "scene_builder":
            return (
                busy,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(),
                gr.update(visible=False, interactive=False),  # asset_save_actor_btn
                gr.update(visible=False, interactive=False),  # asset_save_nonactor_btn
                gr.update(visible=False, interactive=False),  # asset_save_room_btn
                gr.update(visible=False, interactive=False),  # asset_save_general_btn
                False,  # asset_urdf_ready_state
            ) + _step4_hide_all
        return (
            busy,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(),
            gr.update(visible=False, interactive=False),  # asset_save_actor_btn
            gr.update(visible=False, interactive=False),  # asset_save_nonactor_btn
            gr.update(visible=False, interactive=False),  # asset_save_room_btn
            gr.update(visible=False, interactive=False),  # asset_save_general_btn
            False,  # asset_urdf_ready_state
        ) + _step4_hide_all

    def _step4_room_updates(choice: str):
        def _one(chosen: str, label: str):
            return gr.update(visible=True, variant="primary" if chosen == label else "secondary")

        return (
            _one(choice, "极简风"),
            _one(choice, "客厅"),
            _one(choice, "卧室"),
            _one(choice, "餐厅"),
            _one(choice, "卫生间"),
            _one(choice, "儿童房"),
            _one(choice, "书房"),
            gr.update(visible=True),
            gr.update(visible=True),
            choice,
        )

    def _step4_panel_hidden_until_asset_class():
        """资产模式：在用户选定分类并保存前，Step4（房间键 / 日志 / 建立仿真）保持隐藏。"""
        return (gr.update(visible=False),) * 7 + (
            gr.update(visible=False),
            gr.update(visible=False, value="▶️ 建立仿真"),
            gr.update(visible=False),
        )

    def _step4_panel_after_asset_classify():
        """用户点击 Actor/Static/Room/General 并写入 RoboTwin 后，再展开 Step4（先选房间再出建立仿真）。"""
        return (
            gr.update(visible=True, variant="secondary"),
            gr.update(visible=True, variant="secondary"),
            gr.update(visible=True, variant="secondary"),
            gr.update(visible=True, variant="secondary"),
            gr.update(visible=True, variant="secondary", interactive=True),
            gr.update(visible=True, variant="secondary", interactive=True),
            gr.update(visible=True, variant="secondary", interactive=True),
            gr.update(visible=True),
            gr.update(visible=False, value="▶️ 建立仿真"),
            gr.update(visible=False),
        )

    def _save_asset_classify_then_show_step4(mode: str, target_kind: str, request: gr.Request):
        """资产分类按钮：写入 RoboTwin 后再展开 Step4 下游区。"""
        b1, b2, b3, b4 = save_asset_to_robotwin(mode, target_kind, request)
        if mode != "asset_extraction":
            return (b1, b2, b3, b4) + _step4_panel_hidden_until_asset_class()
        return (b1, b2, b3, b4) + _step4_panel_after_asset_classify()

    def _asset_save_actor_chain(mode, request: gr.Request):
        return _save_asset_classify_then_show_step4(mode, "actor", request)

    def _asset_save_nonactor_chain(mode, request: gr.Request):
        return _save_asset_classify_then_show_step4(mode, "non-actor", request)

    def _asset_save_room_chain(mode, request: gr.Request):
        return _save_asset_classify_then_show_step4(mode, "room", request)

    def _asset_save_general_chain(mode, request: gr.Request):
        return _save_asset_classify_then_show_step4(mode, "general", request)

    def _gen_done_ui(mode, asset_urdf_ready, layout_from_step2):
        if mode == "desktop_twin":
            layout = (layout_from_step2 or "").strip() or _latest_dc_layout_json()
        else:
            layout = ""
        btn = gr.update(interactive=True, value="🚀 启动重构/同步")
        rooms_shown = (
            gr.update(visible=True, variant="secondary"),
            gr.update(visible=True, variant="secondary"),
            gr.update(visible=True, variant="secondary"),
            gr.update(visible=True, variant="secondary"),
            gr.update(visible=True, variant="secondary", interactive=True),
            gr.update(visible=True, variant="secondary", interactive=True),
            gr.update(visible=True, variant="secondary", interactive=True),
        )
        if mode == "desktop_twin":
            return (
                btn,
                gr.update(visible=True),                 # dt_success_msg
                *rooms_shown,
                gr.update(visible=True),                 # logs
                gr.update(visible=False, value="▶️ 建立仿真"),  # dc_btn：点选场景后再显示
                gr.update(visible=False),                # stop_btn
                layout,                                  # layout_path_state
                gr.update(visible=False),                # scene_success_msg
                gr.update(visible=False, interactive=False),  # asset_save_actor_btn
                gr.update(visible=False, interactive=False),  # asset_save_nonactor_btn
                gr.update(visible=False, interactive=False),  # asset_save_room_btn
                gr.update(visible=False, interactive=False),  # asset_save_general_btn
                None,                                    # step4_scene_preset
            )
        if mode == "scene_builder":
            # 与 random 一致：不展示 Step4 房间键，布局完成后直接可点「渲染场景」
            rooms_hidden = (gr.update(visible=False),) * 7
            return (
                btn,
                gr.update(visible=False),                # dt_success_msg
                *rooms_hidden,
                gr.update(visible=True),                 # logs
                gr.update(visible=True, value="▶️ 渲染场景"),  # dc_btn
                gr.update(visible=True),                 # stop_btn
                "",                                      # layout_path_state
                gr.update(visible=True),                 # scene_success_msg
                gr.update(visible=False, interactive=False),  # asset_save_actor_btn
                gr.update(visible=False, interactive=False),  # asset_save_nonactor_btn
                gr.update(visible=False, interactive=False),  # asset_save_room_btn
                gr.update(visible=False, interactive=False),  # asset_save_general_btn
                None,                                    # step4_scene_preset
            )
        save_btn_update = gr.update(visible=True, interactive=bool(asset_urdf_ready))
        # Step4 仅在用户点击「资产分类」之一并保存后再展开
        step4_until_class = _step4_panel_hidden_until_asset_class()
        return (
            btn,
            gr.update(visible=False),                # dt_success_msg
            *step4_until_class,
            layout,                                  # layout_path_state
            gr.update(visible=False),                # scene_success_msg
            save_btn_update,                          # asset_save_actor_btn
            save_btn_update,                          # asset_save_nonactor_btn
            save_btn_update,                          # asset_save_room_btn
            save_btn_update,                          # asset_save_general_btn
            None,                                    # step4_scene_preset
        )

    gen_btn.click(
        _gen_start_ui,
        inputs=[current_mode],
        outputs=[
            gen_btn,
            dt_success_msg,
            dt_sync_logs,
            scene_success_msg,
            scene_logs,
            asset_save_actor_btn,
            asset_save_nonactor_btn,
            asset_save_room_btn,
            asset_save_general_btn,
            asset_urdf_ready_state,
            step4_room_raw_btn,
            step4_room_living_btn,
            step4_room_bedroom_btn,
            step4_room_dining_btn,
            step4_room_bath_btn,
            step4_room_kids_btn,
            step4_room_study_btn,
            logs,
            dc_btn,
            stop_btn,
            step4_scene_preset,
        ],
    ).then(
        run_step2_pipeline,
        inputs=[
            current_mode,
            raw_cache,
            image_seg_sam,
            img_dt,
            seed,
            ss_steps,
            slat_steps,
            ss_g,
            slat_g,
            tex_size,
            asset_cat_text,
            height_range_text,
            seg_backend,
            sam3_prompt_state,
            scene_type_state,
            scene_template    
        ],
        outputs=[dt_sync_logs, out_mesh, out_video, scene_logs, asset_urdf_ready_state, layout_path_state],
    ).then(
        _gen_done_ui,
        inputs=[current_mode, asset_urdf_ready_state, layout_path_state],
        outputs=[
            gen_btn,
            dt_success_msg,
            step4_room_raw_btn,
            step4_room_living_btn,
            step4_room_bedroom_btn,
            step4_room_dining_btn,
            step4_room_bath_btn,
            step4_room_kids_btn,
            step4_room_study_btn,
            logs,
            dc_btn,
            stop_btn,
            layout_path_state,
            scene_success_msg,
            asset_save_actor_btn,
            asset_save_nonactor_btn,
            asset_save_room_btn,
            asset_save_general_btn,
            step4_scene_preset,
        ],
    )

    _step4_room_click_outputs = [
        step4_room_raw_btn,
        step4_room_living_btn,
        step4_room_bedroom_btn,
        step4_room_dining_btn,
        step4_room_bath_btn,
        step4_room_kids_btn,
        step4_room_study_btn,
        dc_btn,
        stop_btn,
        step4_scene_preset,
    ]
    step4_room_raw_btn.click(
        lambda: _step4_room_updates("极简风"),
        outputs=_step4_room_click_outputs,
    )
    step4_room_living_btn.click(
        lambda: _step4_room_updates("客厅"),
        outputs=_step4_room_click_outputs,
    )
    step4_room_bedroom_btn.click(
        lambda: _step4_room_updates("卧室"),
        outputs=_step4_room_click_outputs,
    )
    step4_room_dining_btn.click(
        lambda: _step4_room_updates("餐厅"),
        outputs=_step4_room_click_outputs,
    )
    step4_room_bath_btn.click(
        lambda: _step4_room_updates("卫生间"),
        outputs=_step4_room_click_outputs,
    )
    step4_room_kids_btn.click(
        lambda: _step4_room_updates("儿童房"),
        outputs=_step4_room_click_outputs,
    )
    step4_room_study_btn.click(
        lambda: _step4_room_updates("书房"),
        outputs=_step4_room_click_outputs,
    )
    scene_type_living_btn.click(
        lambda: _scene_type_updates("客厅"),
        outputs=[scene_type_living_btn, scene_type_bedroom_btn, scene_type_dining_btn, scene_type_bath_btn, scene_type_children_btn, scene_type_study_btn, scene_type_state],
    )
    scene_type_bedroom_btn.click(
        lambda: _scene_type_updates("卧室"),
        outputs=[scene_type_living_btn, scene_type_bedroom_btn, scene_type_dining_btn, scene_type_bath_btn, scene_type_children_btn, scene_type_study_btn, scene_type_state],
    )
    scene_type_dining_btn.click(
        lambda: _scene_type_updates("餐厅"),
        outputs=[scene_type_living_btn, scene_type_bedroom_btn, scene_type_dining_btn, scene_type_bath_btn, scene_type_children_btn, scene_type_study_btn, scene_type_state],
    )
    scene_type_bath_btn.click(
        lambda: _scene_type_updates("卫生间"),
        outputs=[scene_type_living_btn, scene_type_bedroom_btn, scene_type_dining_btn, scene_type_bath_btn, scene_type_children_btn, scene_type_study_btn, scene_type_state],
    )
    scene_type_children_btn.click(
        lambda: _scene_type_updates("儿童房"),
        outputs=[scene_type_living_btn, scene_type_bedroom_btn, scene_type_dining_btn, scene_type_bath_btn, scene_type_children_btn, scene_type_study_btn, scene_type_state],
    )
    scene_type_study_btn.click(
        lambda: _scene_type_updates("书房"),
        outputs=[scene_type_living_btn, scene_type_bedroom_btn, scene_type_dining_btn, scene_type_bath_btn, scene_type_children_btn, scene_type_study_btn, scene_type_state],
    )

    # Step 4 分发
    dc_btn.click(
        run_step4_dispatch,
        inputs=[current_mode, layout_path_state, step4_scene_preset],
        outputs=logs,
    )
    stop_btn.click(stop_robotwin_bash, outputs=logs)
    asset_save_actor_btn.click(
        _asset_save_actor_chain,
        inputs=[current_mode],
        outputs=[
            asset_save_actor_btn,
            asset_save_nonactor_btn,
            asset_save_room_btn,
            asset_save_general_btn,
            step4_room_raw_btn,
            step4_room_living_btn,
            step4_room_bedroom_btn,
            step4_room_dining_btn,
            step4_room_bath_btn,
            step4_room_kids_btn,
            step4_room_study_btn,
            logs,
            dc_btn,
            stop_btn,
        ],
    )
    asset_save_nonactor_btn.click(
        _asset_save_nonactor_chain,
        inputs=[current_mode],
        outputs=[
            asset_save_actor_btn,
            asset_save_nonactor_btn,
            asset_save_room_btn,
            asset_save_general_btn,
            step4_room_raw_btn,
            step4_room_living_btn,
            step4_room_bedroom_btn,
            step4_room_dining_btn,
            step4_room_bath_btn,
            step4_room_kids_btn,
            step4_room_study_btn,
            logs,
            dc_btn,
            stop_btn,
        ],
    )
    asset_save_room_btn.click(
        _asset_save_room_chain,
        inputs=[current_mode],
        outputs=[
            asset_save_actor_btn,
            asset_save_nonactor_btn,
            asset_save_room_btn,
            asset_save_general_btn,
            step4_room_raw_btn,
            step4_room_living_btn,
            step4_room_bedroom_btn,
            step4_room_dining_btn,
            step4_room_bath_btn,
            step4_room_kids_btn,
            step4_room_study_btn,
            logs,
            dc_btn,
            stop_btn,
        ],
    )
    asset_save_general_btn.click(
        _asset_save_general_chain,
        inputs=[current_mode],
        outputs=[
            asset_save_actor_btn,
            asset_save_nonactor_btn,
            asset_save_room_btn,
            asset_save_general_btn,
            step4_room_raw_btn,
            step4_room_living_btn,
            step4_room_bedroom_btn,
            step4_room_dining_btn,
            step4_room_bath_btn,
            step4_room_kids_btn,
            step4_room_study_btn,
            logs,
            dc_btn,
            stop_btn,
        ],
    )

    demo.load(start_session); demo.unload(end_session)

if __name__ == "__main__":
    demo.launch(server_port=8081)
