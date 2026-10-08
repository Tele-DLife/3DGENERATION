"""Standalone ranked-friction URDF flow.

This file intentionally contains the complete "new flow":
1) prompt template with ranked friction guidance
2) generator class with compatible I/O
3) optional height sanity re-check using mesh + 10cm cube + VLM review
"""

from __future__ import annotations

import json
import logging
import os
import re

import numpy as np
import trimesh
from PIL import Image

from embodied_gen.utils.tags import VERSION
from embodied_gen.validators.urdf_convertor import URDFGenerator

__all__ = [
    "build_urdf_prompt_template_ranked_friction",
    "URDFGeneratorRankedFriction",
]

logger = logging.getLogger(__name__)


def build_urdf_prompt_template_ranked_friction(view_desc: str) -> str:
    """Build a prompt compatible with the existing URDF parser output format.

    The output labels/order are kept the same as current parser expectations:
    Category, Description, Pose, Height, Weight, Static friction coefficient,
    Dynamic friction coefficient, Color, RGBA.
    """
    color_list_str = (
        "red, orange, yellow, green, blue, purple, pink, black, white, grey, "
        "brown, silver, golden, copper"
    )

    return (
        view_desc
        + f"""of the 3D object asset,
            category: {{category}}.
            You are an expert in 3D object analysis and physical property estimation.
            Give the category of this object asset (within 3 words), (if category is
            already provided, use it directly), accurately describe this 3D object asset (within 15 words),
            If the provided category hint is a long phrase with adjectives, locations, colors, materials,
            or other modifiers, normalize it to the head object noun category only (e.g., "the green marker on table" -> "marker").
            Determine the pose of the object in the first image based on all views and estimate the true vertical height
            (vertical projection) range of the object (in meters), i.e., how tall the object appears from top
            to bottom in the first image. also weight range (unit: kilogram), the average
            static friction coefficient of the object relative to rubber and the average dynamic friction
            coefficient of the object relative to rubber. Additionally, identify the main color of the object and its RGBA value:
            1. Select one color name from this list: [{color_list_str}].
            2. Provide an estimated RGBA color value (0.0-1.0 for each component).
            Return response in format as shown in Output Example.

            Friction estimation method (only for friction; keep all other fields unchanged):
            - First classify the object into one reference friction rank by closest material/surface behavior vs rubber.
            - Ranks are ordered from LOW to HIGH friction:
              1) Smooth metal laptop shell (very low)
              2) Glass / smooth hard plastic bottle (low)
              3) Apple peel / waxy fruit skin (low-medium)
              4) Coated paper/cardboard book cover (medium)
              5) Ceramic mug / glazed pottery (medium+)
              6) Unfinished wood / matte painted hard surface (medium-high)
              7) Rubber-coated object / silicone case (high)
              8) Plush toy / fuzzy fabric surface (very high)
            - Then map the chosen rank to friction values (relative to rubber):
              rank 1: static 0.20-0.35, dynamic 0.15-0.30
              rank 2: static 0.30-0.50, dynamic 0.25-0.45
              rank 3: static 0.40-0.60, dynamic 0.35-0.55
              rank 4: static 0.50-0.70, dynamic 0.45-0.65
              rank 5: static 0.60-0.80, dynamic 0.50-0.70
              rank 6: static 0.70-0.95, dynamic 0.60-0.85
              rank 7: static 0.90-1.20, dynamic 0.75-1.05
              rank 8: static 1.10-1.60, dynamic 0.90-1.35
            - Choose a concrete static/dynamic value inside the selected rank range.
            - Static friction must be >= dynamic friction.
            - Do NOT output rank number or reasoning; only output the two friction lines in the required format.

            Output Example:
            Category: cup
            Description: shiny golden cup with floral design
            Pose: <short_description_within_10_words>
            Height: 0.10-0.15 m
            Weight: 0.3-0.6 kg
            Static friction coefficient: 0.6
            Dynamic friction coefficient: 0.5
            Color: red
            RGBA: [1.0, 0.0, 0.0, 1.0]


            IMPORTANT: Estimating Vertical Height from the First (Front View) Image and pose estimation based on all views.
            - The "vertical height" refers to the real-world vertical size of the object
            as projected in the first image, aligned with the image's vertical axis.
            - For flat objects like plates or disks or book, if their face is visible in the front view,
            use the diameter as the vertical height. If the edge is visible, use the thickness instead.
            - This is not necessarily the full length of the object, but how tall it appears
            in the first image vertically, based on its pose and orientation estimation on all views.
            - Distinguish whether the entire objects such as plates, books, pens, spoons, fork are placed
                horizontally or vertically based on pictures from left, right views.

            Estimate the vertical projection of their real length based on its pose.
            For example:
              - A pen standing upright in the first image (aligned with the image's vertical axis)
                full body visible in the first and other image: -> vertically -> vertical height ~= 0.14-0.20 m
              - A pen lying flat in the first image or either the tip or the tail is facing the image
                (showing thickness or as a circle), left/right view can show the full body
                -> horizontally -> vertical height ~= 0.018-0.025 m
              - Tilted pen in the first image (e.g., ~45 degree): vertical height ~= 0.07-0.12 m
            - Use the rest views to help determine the object's 3D pose and orientation.
            Assume the object is in real-world scale and estimate the approximate vertical height
            based on the pose estimation and how large it appears vertically in the first image.
            """
    )


class URDFGeneratorRankedFriction(URDFGenerator):
    """Drop-in replacement generator with ranked friction + height VLM refine."""

    def __init__(
        self,
        gpt_client,
        mesh_file_list: list[str] = ["material_0.png", "material.mtl"],
        prompt_template: str = None,
        attrs_name: list[str] = None,
        render_dir: str = "urdf_renders",
        render_view_num: int = 4,
        decompose_convex: bool = False,
        rotate_xyzw: list[float] = (0.7071, 0, 0, 0.7071),
        enable_height_vlm_refine: bool = True,
        reference_cube_size_m: float = 0.10,
    ) -> None:
        if render_view_num == 4:
            view_desc = "This is an orthographic projection showing the front(1st image), right(2nd), back(3rd), and left(4th) views."  # noqa
        else:
            view_desc = "This is the rendered views "

        if prompt_template is None:
            prompt_template = build_urdf_prompt_template_ranked_friction(view_desc)

        super().__init__(
            gpt_client=gpt_client,
            mesh_file_list=mesh_file_list,
            prompt_template=prompt_template,
            attrs_name=attrs_name,
            render_dir=render_dir,
            render_view_num=render_view_num,
            decompose_convex=decompose_convex,
            rotate_xyzw=rotate_xyzw,
        )
        self.enable_height_vlm_refine = enable_height_vlm_refine
        self.reference_cube_size_m = reference_cube_size_m

    @staticmethod
    def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        z_axis = eye - target
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-8:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            z_axis /= z_norm

        x_axis = np.cross(up, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            x_axis /= x_norm
        y_axis = np.cross(z_axis, x_axis)

        pose = np.eye(4, dtype=float)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = eye
        return pose

    def _render_height_reference_snapshot(
        self,
        mesh_path: str,
        output_root: str,
        object_height_m: float,
        cube_size_m: float = 0.10,
    ) -> str | None:
        try:
            os.environ.setdefault("PYRENDER_BACKEND", "egl")
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            import pyrender
        except Exception as e:
            logger.warning(f"Skip height sanity check: pyrender unavailable: {e}")
            return None

        try:
            loaded = trimesh.load(mesh_path)
            if isinstance(loaded, trimesh.Scene):
                parts = [
                    g.copy()
                    for g in loaded.dump()
                    if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0
                ]
                if not parts:
                    return None
                mesh = trimesh.util.concatenate(parts)
            else:
                mesh = loaded.copy()
            if len(mesh.vertices) == 0:
                return None

            max_extent = float(np.ptp(mesh.vertices, axis=0).max())
            if max_extent < 1e-8:
                return None
            mesh.vertices /= max_extent

            raw_height = float(np.ptp(mesh.vertices, axis=0)[1])
            if raw_height < 1e-8:
                return None
            mesh.apply_scale(float(object_height_m) / raw_height)

            obj_bounds = mesh.bounds.copy()
            mesh.apply_translation([0.0, -obj_bounds[0, 1], 0.0])
            obj_bounds = mesh.bounds.copy()

            # Side-by-side on the ground (Y-up), gap along X; camera from +Z for a clear
            # front view so the cube does not sit beside the object in depth (less occlusion).
            gap = max(0.04, 0.25 * float(cube_size_m))
            dx = -gap / 2.0 - float(obj_bounds[1, 0])
            mesh.apply_translation([dx, 0.0, 0.0])

            cube = trimesh.creation.box(
                extents=[float(cube_size_m), float(cube_size_m), float(cube_size_m)]
            )
            cube_cx = gap / 2.0 + float(cube_size_m) / 2.0
            cube.apply_translation([cube_cx, float(cube_size_m) / 2.0, 0.0])

            obj_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
            cube_mesh = pyrender.Mesh.from_trimesh(cube, smooth=False)

            scene = pyrender.Scene(
                bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
                ambient_light=np.array([0.35, 0.35, 0.35], dtype=float),
            )
            scene.add(obj_mesh)
            scene.add(cube_mesh)

            bounds = np.array(
                [
                    np.minimum(mesh.bounds[0], cube.bounds[0]),
                    np.maximum(mesh.bounds[1], cube.bounds[1]),
                ]
            )
            center = (bounds[0] + bounds[1]) / 2.0
            extents = bounds[1] - bounds[0]
            span_x = float(extents[0])
            span_y = float(extents[1])
            span_z = float(extents[2])
            depth = max(span_x, span_y, span_z, cube_size_m, 0.15)
            # Front view: camera on +Z, slight elevation on Y, look at scene center.
            eye = center + np.array(
                [0.0, max(0.12, 0.35 * span_y), max(0.45, 2.0 * depth)],
                dtype=float,
            )
            cam_pose = self._look_at(eye=eye, target=center)

            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            scene.add(camera, pose=cam_pose)
            scene.add(
                pyrender.DirectionalLight(color=np.ones(3), intensity=4.0),
                pose=cam_pose,
            )

            out_dir = os.path.join(output_root, "height_sanity_check")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "mesh_with_reference_cube.png")

            renderer = pyrender.OffscreenRenderer(
                viewport_width=1024, viewport_height=768
            )
            try:
                color, _ = renderer.render(scene)
            finally:
                renderer.delete()

            Image.fromarray(color).save(out_path)
            return out_path
        except Exception as e:
            logger.warning(f"Failed to render height sanity snapshot: {e}")
            return None

    @staticmethod
    def _parse_vlm_height_review_response(response: str) -> dict | None:
        if not response:
            return None
        candidates = [response.strip()]
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            candidates.insert(0, match.group(0))
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    def _refine_height_with_vlm(
        self,
        mesh_path: str,
        output_root: str,
        category: str,
        min_height: float,
        max_height: float,
    ) -> tuple[float, float]:
        if min_height <= 0 or max_height <= 0 or max_height < min_height:
            return min_height, max_height

        current_mid = (min_height + max_height) / 2.0
        snapshot_path = self._render_height_reference_snapshot(
            mesh_path=mesh_path,
            output_root=output_root,
            object_height_m=current_mid,
            cube_size_m=self.reference_cube_size_m,
        )
        if snapshot_path is None:
            return min_height, max_height

        logger.info(
            "Height sanity VLM: snapshot=%s, range=%.4f-%.4f m, mid=%.4f m",
            snapshot_path,
            min_height,
            max_height,
            current_mid,
        )

        review_prompt = f"""
You sanity-check HEIGHT (meters) using ONE snapshot image only.

Scene layout (trust this geometry):
- FRONT view: target object on the LEFT, reference cube on the RIGHT, on the same ground.
- The cube is an axis-aligned box with edge length EXACTLY {self.reference_cube_size_m:.2f} m in world space.
- The object mesh has ALREADY been uniformly scaled so that, if the first-stage height estimate were correct,
  its vertical extent (Y axis in the scene) would match midpoint {current_mid:.4f} m (range {min_height:.4f}-{max_height:.4f} m).

A previous vision-language model (multiview) produced that height range; this image is only a coarse check.

Task:
- Compare the APPARENT VERTICAL SIZE of the object vs the cube in THIS image (ignore horizontal width unless the object is clearly lying down).
- Set is_reasonable=false ONLY if the side-by-side snapshot CLEARLY contradicts the stated range (e.g. object's visible vertical span is obviously far below or far above the cube's edge length in a way that cannot be explained by pose/occlusion).
- If uncertain, lighting is poor, mesh is ambiguous, or agreement is within ~30%, set is_reasonable=true.
- NEVER claim the object is "shorter than the cube" if the object's vertical span in the image is visibly TALLER than the cube's vertical span.

Object category hint: {category or "unknown"}.

Return STRICT JSON only (no markdown/code fence):
{{
  "is_reasonable": true or false,
  "suggested_height_m": <float>,
  "confidence": <float between 0 and 1>,
  "reason": "<short reason>"
}}
If is_reasonable is true, suggested_height_m should be close to {current_mid:.4f}.
""".strip()

        response = self.gpt_client.query(review_prompt, [snapshot_path])
        logger.info(
            "Height sanity VLM raw response (full):\n%s",
            response if response is not None else "<None>",
        )
        parsed = self._parse_vlm_height_review_response(response or "")
        if not parsed:
            logger.warning(
                "Height sanity VLM returned unparseable response (raw was logged above)."
            )
            return min_height, max_height

        is_reasonable = bool(parsed.get("is_reasonable", True))
        suggested = parsed.get("suggested_height_m", current_mid)
        try:
            suggested = float(suggested)
        except Exception:
            return min_height, max_height

        if is_reasonable:
            return min_height, max_height

        if not np.isfinite(suggested) or suggested <= 0:
            return min_height, max_height

        ratio = suggested / max(current_mid, 1e-6)
        if ratio < 0.25 or ratio > 4.0:
            logger.warning(
                "Ignore VLM height suggestion due to extreme ratio. "
                f"suggested={suggested:.4f}, current={current_mid:.4f}"
            )
            return min_height, max_height

        span = max_height - min_height
        if span <= 1e-4:
            new_min = max(1e-4, suggested * 0.9)
            new_max = max(new_min + 1e-4, suggested * 1.1)
        else:
            half = span / 2.0
            new_min = max(1e-4, suggested - half)
            new_max = max(new_min + 1e-4, suggested + half)

        logger.info(
            "Height refined by VLM: %.4f-%.4f -> %.4f-%.4f (reason=%s)",
            min_height,
            max_height,
            new_min,
            new_max,
            str(parsed.get("reason", ""))[:120],
        )
        return round(new_min, 4), round(new_max, 4)

    def __call__(
        self,
        mesh_path: str,
        output_root: str,
        text_prompt: str = None,
        category: str = "unknown",
        **kwargs,
    ):
        base_prompt = self.prompt_template.format(category=category.lower())
        hint = (text_prompt or "").strip()
        if hint:
            full_prompt = (
                base_prompt
                + "\n\n---\nUser/segmentation text hint (disambiguation only; "
                "you must still follow the Output Example format line-for-line):\n"
                + hint
            )
        else:
            full_prompt = base_prompt

        from embodied_gen.utils.process_media import render_asset3d

        image_path = render_asset3d(
            mesh_path,
            output_root,
            num_images=self.render_view_num,
            output_subdir=self.output_render_dir,
            no_index_file=True,
        )
        response = self.gpt_client.query(full_prompt, image_path)
        logger.info(
            "First-pass attribute LLM raw response (full, multiview):\n%s",
            response if response is not None else "<None>",
        )
        if response is None:
            asset_attrs = self._default_asset_attrs(category)
        else:
            try:
                asset_attrs = self.parse_response(response)
            except Exception as e:
                logger.warning(
                    "Failed to parse GPT response, fallback to defaults. error=%s, response=%r",
                    e,
                    response[:500],
                )
                asset_attrs = self._default_asset_attrs(category)

        for key in self.attrs_name:
            if key in kwargs:
                asset_attrs[key] = kwargs[key]

        forced_category = (category or "").lstrip().lower()
        if forced_category and forced_category != "unknown":
            asset_attrs["category"] = forced_category

        if self.enable_height_vlm_refine:
            refined_min_h, refined_max_h = self._refine_height_with_vlm(
                mesh_path=mesh_path,
                output_root=output_root,
                category=asset_attrs.get("category", forced_category),
                min_height=float(asset_attrs["min_height"]),
                max_height=float(asset_attrs["max_height"]),
            )
            asset_attrs["min_height"] = refined_min_h
            asset_attrs["max_height"] = refined_max_h

        asset_attrs["real_height"] = round(
            (asset_attrs["min_height"] + asset_attrs["max_height"]) / 2, 4
        )
        asset_attrs["version"] = kwargs.get("version", asset_attrs.get("version", VERSION))

        self.estimated_attrs = self.get_estimated_attributes(asset_attrs)
        urdf_path = self.generate_urdf(mesh_path, output_root, asset_attrs)
        logger.info("URDF written (first-pass LLM text was logged earlier): %s", urdf_path)
        return urdf_path
