# Project 3DGENERATION
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
# Modifications Copyright (c) 2025-2026 3DGENERATION Contributors.
# This file contains modifications to EmbodiedGen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import spaces


def _prefer_conda_libstdcpp():
    """Ensure conda's libstdc++ is preferred over system libs."""
    conda_prefix = os.getenv("CONDA_PREFIX")
    if not conda_prefix:
        return
    conda_lib = os.path.join(conda_prefix, "lib")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld_path:
        os.environ["LD_LIBRARY_PATH"] = conda_lib
        return
    paths = ld_path.split(":")
    if conda_lib not in paths:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_path}"


_prefer_conda_libstdcpp()

# Keep runtime caches aligned with scripts/prefetch_models.py.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache_root = os.path.join(_repo_root, "models", "cache")
_hf_home = os.path.join(_cache_root, "hf")
os.environ.setdefault("HF_HOME", _hf_home)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_hf_home, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_hf_home, "hub"))
os.environ.setdefault("TORCH_HOME", os.path.join(_cache_root, "torch"))
os.environ.setdefault("U2NET_HOME", os.path.join(_cache_root, "u2net"))

import gc
import logging
import shutil
import sys
from glob import glob

import cv2
import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image
from embodied_gen.data.backproject_v3 import entrypoint as backproject_api_v3
from embodied_gen.data.utils import trellis_preprocess, zip_files
from embodied_gen.models.gs_model import GaussianOperator
from embodied_gen.models.sam3d import Sam3dInference
from embodied_gen.models.segment_model import (
    SAMPredictor,
)
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.process_media import (
    filter_image_small_connected_components,
    merge_images_video,
)
from embodied_gen.utils.tags import VERSION
from embodied_gen.utils.trender import pack_state, render_video, unpack_state
from embodied_gen.validators.quality_checkers import (
    BaseChecker,
    ImageAestheticChecker,
    ImageSegChecker,
    MeshGeoChecker,
)
from embodied_gen.validators.urdf_convertor import URDFGenerator as URDFGeneratorDefault

try:
    from embodied_gen.validators.urdf_converter_CoT import (
        URDFGeneratorRankedFriction,
    )
except Exception:
    URDFGeneratorRankedFriction = None

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_dir, ".."))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

MAX_SEED = 100000


def _select_urdf_generator():
    """Select URDF generator by environment switch."""
    mode_raw = os.getenv("EMBODIEDGEN_URDF_CONVERTER", "default")
    mode = (mode_raw or "").strip().lower()
    if mode == "cot":
        if URDFGeneratorRankedFriction is not None:
            return URDFGeneratorRankedFriction, "cot"
        logger.warning(
            "EMBODIEDGEN_URDF_CONVERTER=%s but CoT generator import failed; fallback to default.",
            mode_raw,
        )
    return URDFGeneratorDefault, "default"

app_name = os.getenv("GRADIO_APP", "imageto3d_sam3d")
_use_sam3d = "sam3d" in app_name
if not _use_sam3d:
    from embodied_gen.utils.monkey_patches import monkey_path_trellis
    from thirdparty.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
    from thirdparty.TRELLIS.trellis.utils import postprocessing_utils

    monkey_path_trellis()
if app_name.startswith("imageto3d"):
    SAM_PREDICTOR = SAMPredictor(model_type="vit_h", device="cpu")
    if _use_sam3d:
        PIPELINE = Sam3dInference()
    else:
        PIPELINE = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        # PIPELINE.cuda()
    SEG_CHECKER = ImageSegChecker(GPT_CLIENT)
    GEO_CHECKER = MeshGeoChecker(GPT_CLIENT)
    AESTHETIC_CHECKER = ImageAestheticChecker()
    CHECKERS = [GEO_CHECKER, SEG_CHECKER, AESTHETIC_CHECKER]
    TMP_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sessions/imageto3d"
    )
    os.makedirs(TMP_DIR, exist_ok=True)
else:
    raise ValueError(f"Unsupported GRADIO_APP mode: {app_name}")


def start_session(req: gr.Request) -> None:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request) -> None:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)


def active_btn_by_content(content: gr.Image) -> gr.Button:
    interactive = True if content is not None else False

    return gr.Button(interactive=interactive)


def select_point(
    image: np.ndarray,
    sel_pix: list,
    point_type: str,
    evt: gr.SelectData,
):
    # Only (re)initialize SAM predictor if it's not already set to avoid lag.
    if not getattr(SAM_PREDICTOR.predictor, "is_image_set", False):
        if isinstance(image, np.ndarray):
            sam_image = SAM_PREDICTOR.preprocess_image(image)
            SAM_PREDICTOR.predictor.set_image(sam_image)
            image = sam_image

    if point_type == "foreground_point":
        sel_pix.append((evt.index, 1))  # append the foreground_point
    elif point_type == "background_point":
        sel_pix.append((evt.index, 0))  # append the background_point
    else:
        sel_pix.append((evt.index, 1))  # default foreground_point

    # Keep an untouched copy for mask generation/export.
    base_image = image.copy() if isinstance(image, np.ndarray) else image
    masks = SAM_PREDICTOR.generate_masks(base_image, sel_pix)
    seg_image = SAM_PREDICTOR.get_segmented_image(base_image, masks)

    # Draw interactive click markers only on preview image.
    preview_image = (
        base_image.copy() if isinstance(base_image, np.ndarray) else base_image
    )

    for point, label in sel_pix:
        color = (255, 0, 0) if label == 0 else (0, 255, 0)
        marker_type = 1 if label == 0 else 5
        cv2.drawMarker(
            preview_image,
            point,
            color,
            markerType=marker_type,
            markerSize=15,
            thickness=10,
        )

    torch.cuda.empty_cache()

    return (preview_image, masks), seg_image


@spaces.GPU
def image_to_3d(
    image: Image.Image,
    seed: int,
    ss_sampling_steps: int,
    slat_sampling_steps: int,
    raw_image_cache: Image.Image,
    ss_guidance_strength: float,
    slat_guidance_strength: float,
    sam_image: Image.Image = None,
    is_sam_image: bool = False,
    req: gr.Request = None,
) -> tuple[dict, str]:
    if is_sam_image:
        seg_image = filter_image_small_connected_components(sam_image)
        seg_image = Image.fromarray(seg_image, mode="RGBA")
    else:
        seg_image = image

    if isinstance(seg_image, np.ndarray):
        seg_image = Image.fromarray(seg_image)

    if isinstance(PIPELINE, Sam3dInference):
        outputs = PIPELINE.run(
            seg_image,
            seed=seed,
            stage1_inference_steps=ss_sampling_steps,
            stage2_inference_steps=slat_sampling_steps,
        )
    else:
        PIPELINE.cuda()
        seg_image = trellis_preprocess(seg_image)
        outputs = PIPELINE.run(
            seg_image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        # Set back to cpu for memory saving.
        PIPELINE.cpu()

    gs_model = outputs["gaussian"][0]
    mesh_model = outputs["mesh"][0]
    color_images = render_video(gs_model, r=1.85)["color"]
    normal_images = render_video(mesh_model, r=1.85)["normal"]

    output_root = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(output_root, exist_ok=True)
    seg_image.save(f"{output_root}/seg_image.png")
    raw_image_cache.save(f"{output_root}/raw_image.png")

    video_path = os.path.join(output_root, "gs_mesh.mp4")
    merge_images_video(color_images, normal_images, video_path)
    state = pack_state(gs_model, mesh_model)

    gc.collect()
    torch.cuda.empty_cache()

    return state, video_path


def extract_3d_representations_v3(
    state: dict,
    texture_size: int,
    req: gr.Request = None,
):
    """Back-Projection Version with Optimization-Based."""
    output_root = TMP_DIR
    if req is not None:
        output_root = os.path.join(output_root, str(req.session_hash))
    user_dir = output_root
    gs_model, mesh_model = unpack_state(state, device="cpu")

    filename = "sample"
    gs_path = os.path.join(user_dir, f"{filename}_gs.ply")
    gs_model.save_ply(gs_path)

    # Rotate mesh and GS by 90 degrees around Z-axis.
    rot_matrix = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    gs_add_rot = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    mesh_add_rot = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    # Addtional rotation for GS to align mesh.
    gs_rot = np.array(gs_add_rot) @ np.array(rot_matrix)
    pose = GaussianOperator.trans_to_quatpose(gs_rot)
    aligned_gs_path = gs_path.replace(".ply", "_aligned.ply")
    GaussianOperator.resave_ply(
        in_ply=gs_path,
        out_ply=aligned_gs_path,
        instance_pose=pose,
        device="cpu",
    )

    mesh = trimesh.Trimesh(
        vertices=mesh_model.vertices.cpu().numpy(),
        faces=mesh_model.faces.cpu().numpy(),
    )
    mesh.vertices = mesh.vertices @ np.array(mesh_add_rot)
    mesh.vertices = mesh.vertices @ np.array(rot_matrix)

    mesh_obj_path = os.path.join(user_dir, f"{filename}.obj")
    mesh.export(mesh_obj_path)

    mesh = backproject_api_v3(
        gs_path=aligned_gs_path,
        mesh_path=mesh_obj_path,
        output_path=mesh_obj_path,
        skip_fix_mesh=False,
        texture_size=texture_size,
    )

    mesh_glb_path = os.path.join(user_dir, f"{filename}.glb")
    mesh.export(mesh_glb_path)

    return mesh_glb_path, gs_path, mesh_obj_path, aligned_gs_path


def extract_urdf(
    gs_path: str,
    mesh_obj_path: str,
    asset_cat_text: str,
    height_range_text: str,
    mass_range_text: str,
    asset_version_text: str,
    text_prompt: str = "",
    req: gr.Request = None,
):
    output_root = TMP_DIR
    if req is not None:
        output_root = os.path.join(output_root, str(req.session_hash))

    # Convert to URDF and recover attrs by GPT.
    filename = "sample"
    urdf_generator_cls, urdf_generator_mode = _select_urdf_generator()
    logger.info("Using URDF generator mode: %s", urdf_generator_mode)
    urdf_convertor = urdf_generator_cls(
        GPT_CLIENT, render_view_num=4, decompose_convex=True
    )
    asset_attrs = {
        "version": VERSION,
        "gs_model": f"{urdf_convertor.output_mesh_dir}/{filename}_gs.ply",
    }
    if asset_version_text:
        asset_attrs["version"] = asset_version_text
    if asset_cat_text:
        asset_attrs["category"] = asset_cat_text.lower()
    if height_range_text:
        try:
            min_height, max_height = map(float, height_range_text.split("-"))
            asset_attrs["min_height"] = min_height
            asset_attrs["max_height"] = max_height
        except ValueError:
            return "Invalid height input format. Use the format: min-max."
    if mass_range_text:
        try:
            min_mass, max_mass = map(float, mass_range_text.split("-"))
            asset_attrs["min_mass"] = min_mass
            asset_attrs["max_mass"] = max_mass
        except ValueError:
            return "Invalid mass input format. Use the format: min-max."

    urdf_kwargs = dict(asset_attrs)
    urdf_kwargs["text_prompt"] = (text_prompt or "").strip()

    urdf_path = urdf_convertor(
        mesh_path=mesh_obj_path,
        output_root=f"{output_root}/URDF_{filename}",
        **urdf_kwargs,
    )

    # Rescale GS and save to URDF/mesh folder.
    real_height = urdf_convertor.get_attr_from_urdf(
        urdf_path, attr_name="real_height"
    )
    out_gs = f"{output_root}/URDF_{filename}/{urdf_convertor.output_mesh_dir}/{filename}_gs.ply"  # noqa
    GaussianOperator.resave_ply(
        in_ply=gs_path,
        out_ply=out_gs,
        real_height=real_height,
        device="cpu",
    )

    # Quality check and update .urdf file.
    mesh_out = f"{output_root}/URDF_{filename}/{urdf_convertor.output_mesh_dir}/{filename}.obj"  # noqa
    trimesh.load(mesh_out).export(mesh_out.replace(".obj", ".glb"))
    # image_paths = render_asset3d(
    #     mesh_path=mesh_out,
    #     output_root=f"{output_root}/URDF_{filename}",
    #     output_subdir="qa_renders",
    #     num_images=8,
    #     elevation=(30, -30),
    #     distance=5.5,
    # )

    image_dir = f"{output_root}/URDF_{filename}/{urdf_convertor.output_render_dir}/image_color"  # noqa
    image_paths = glob(f"{image_dir}/*.png")
    images_list = []
    for checker in CHECKERS:
        images = image_paths
        if isinstance(checker, ImageSegChecker):
            if req is not None:
                images = [
                    f"{TMP_DIR}/{req.session_hash}/raw_image.png",
                    f"{TMP_DIR}/{req.session_hash}/seg_image.png",
                ]
            else:
                images = []
        images_list.append(images)

    results = BaseChecker.validate(CHECKERS, images_list)
    urdf_convertor.add_quality_tag(urdf_path, results)

    # Zip urdf files
    urdf_zip = zip_files(
        input_paths=[
            f"{output_root}/URDF_{filename}/{urdf_convertor.output_mesh_dir}",
            f"{output_root}/URDF_{filename}/{filename}.urdf",
        ],
        output_zip=f"{output_root}/urdf_{filename}.zip",
    )

    estimated_type = urdf_convertor.estimated_attrs["category"]
    estimated_height = urdf_convertor.estimated_attrs["height"]
    estimated_mass = urdf_convertor.estimated_attrs["mass"]
    estimated_mu = urdf_convertor.estimated_attrs["mu"]

    return (
        urdf_zip,
        estimated_type,
        estimated_height,
        estimated_mass,
        estimated_mu,
    )
