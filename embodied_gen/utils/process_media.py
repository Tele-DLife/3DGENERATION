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


import logging
import math
import os
from glob import glob
from typing import Union

import cv2
import imageio
import numpy as np
import spaces
from moviepy.editor import VideoFileClip, clips_array
from PIL import Image
from embodied_gen.data.differentiable_render import entrypoint as render_api

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    "render_asset3d",
    "merge_images_video",
    "filter_small_connected_components",
    "filter_image_small_connected_components",
    "combine_images_to_grid",
    "check_object_edge_truncated",
    "vcat_pil_images",
]


@spaces.GPU
def render_asset3d(
    mesh_path: str,
    output_root: str,
    distance: float = 5.0,
    num_images: int = 1,
    elevation: list[float] = (0.0,),
    pbr_light_factor: float = 1.2,
    return_key: str = "image_color/*",
    output_subdir: str = "renders",
    gen_color_mp4: bool = False,
    gen_viewnormal_mp4: bool = False,
    gen_glonormal_mp4: bool = False,
    no_index_file: bool = False,
    with_mtl: bool = True,
) -> list[str]:
    """Renders a 3D mesh asset and returns output image paths.

    Args:
        mesh_path (str): Path to the mesh file.
        output_root (str): Directory to save outputs.
        distance (float, optional): Camera distance.
        num_images (int, optional): Number of views to render.
        elevation (list[float], optional): Camera elevation angles.
        pbr_light_factor (float, optional): PBR lighting factor.
        return_key (str, optional): Glob pattern for output images.
        output_subdir (str, optional): Subdirectory for outputs.
        gen_color_mp4 (bool, optional): Generate color MP4 video.
        gen_viewnormal_mp4 (bool, optional): Generate view normal MP4.
        gen_glonormal_mp4 (bool, optional): Generate global normal MP4.
        no_index_file (bool, optional): Skip index file saving.
        with_mtl (bool, optional): Use mesh material.

    Returns:
        list[str]: List of output image file paths.

    Example:
        ```py
        from embodied_gen.utils.process_media import render_asset3d

        image_paths = render_asset3d(
            mesh_path="path_to_mesh.obj",
            output_root="path_to_save_dir",
            num_images=4,
            elevation=(30, -30),
            output_subdir="renders",
            no_index_file=True,
        )
        ```
    """
    input_args = dict(
        mesh_path=mesh_path,
        output_root=output_root,
        uuid=output_subdir,
        distance=distance,
        num_images=num_images,
        elevation=elevation,
        pbr_light_factor=pbr_light_factor,
        with_mtl=with_mtl,
        gen_color_mp4=gen_color_mp4,
        gen_viewnormal_mp4=gen_viewnormal_mp4,
        gen_glonormal_mp4=gen_glonormal_mp4,
        no_index_file=no_index_file,
    )

    try:
        _ = render_api(**input_args)
    except Exception as e:
        logger.error(f"Error occurred during rendering: {e}.")

    dst_paths = glob(os.path.join(output_root, output_subdir, return_key))

    return dst_paths


def merge_images_video(color_images, normal_images, output_path) -> None:
    """Merges color and normal images into a video.

    Args:
        color_images (list[np.ndarray]): List of color images.
        normal_images (list[np.ndarray]): List of normal images.
        output_path (str): Path to save the output video.
    """
    width = color_images[0].shape[1]
    combined_video = [
        np.hstack([rgb_img[:, : width // 2], normal_img[:, width // 2 :]])
        for rgb_img, normal_img in zip(color_images, normal_images)
    ]
    imageio.mimsave(output_path, combined_video, fps=50)

    return


def merge_video_video(
    video_path1: str, video_path2: str, output_path: str
) -> None:
    """Merges two videos by combining their left and right halves.

    Args:
        video_path1 (str): Path to first video.
        video_path2 (str): Path to second video.
        output_path (str): Path to save the merged video.
    """
    clip1 = VideoFileClip(video_path1)
    clip2 = VideoFileClip(video_path2)

    if clip1.size != clip2.size:
        raise ValueError("The resolutions of the two videos do not match.")

    width, height = clip1.size
    clip1_half = clip1.crop(x1=0, y1=0, x2=width // 2, y2=height)
    clip2_half = clip2.crop(x1=width // 2, y1=0, x2=width, y2=height)
    final_clip = clips_array([[clip1_half, clip2_half]])
    final_clip.write_videofile(output_path, codec="libx264")


def filter_small_connected_components(
    mask: Union[Image.Image, np.ndarray],
    area_ratio: float,
    connectivity: int = 8,
) -> np.ndarray:
    """Removes small connected components from a binary mask.

    Args:
        mask (Union[Image.Image, np.ndarray]): Input mask.
        area_ratio (float): Minimum area ratio for components.
        connectivity (int, optional): Connectivity for labeling.

    Returns:
        np.ndarray: Mask with small components removed.
    """
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=connectivity,
    )

    small_components = np.zeros_like(mask, dtype=np.uint8)
    mask_area = (mask != 0).sum()
    min_area = mask_area // area_ratio
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            small_components[labels == label] = 255

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(small_components))

    return mask


def filter_image_small_connected_components(
    image: Union[Image.Image, np.ndarray],
    area_ratio: float = 10,
    connectivity: int = 8,
) -> np.ndarray:
    """Removes small connected components from the alpha channel of an image.

    Args:
        image (Union[Image.Image, np.ndarray]): Input image.
        area_ratio (float, optional): Minimum area ratio.
        connectivity (int, optional): Connectivity for labeling.

    Returns:
        np.ndarray: Image with filtered alpha channel.
    """
    if isinstance(image, Image.Image):
        image = image.convert("RGBA")
        image = np.array(image)

    mask = image[..., 3]
    mask = filter_small_connected_components(mask, area_ratio, connectivity)
    image[..., 3] = mask

    return image


def keep_largest_connected_component(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")

    img_arr = np.array(pil_img)
    alpha_channel = img_arr[:, :, 3]

    _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    if num_labels < 2:
        return pil_img

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    new_alpha = np.where(labels == largest_label, alpha_channel, 0).astype(
        np.uint8
    )
    img_arr[:, :, 3] = new_alpha

    return Image.fromarray(img_arr)


def combine_images_to_grid(
    images: list[str | Image.Image],
    cat_row_col: tuple[int, int] = None,
    target_wh: tuple[int, int] = (512, 512),
    image_mode: str = "RGB",
) -> list[Image.Image]:
    """Combines multiple images into a grid.

    Args:
        images (list[str | Image.Image]): List of image paths or PIL Images.
        cat_row_col (tuple[int, int], optional): Grid rows and columns.
        target_wh (tuple[int, int], optional): Target image size.
        image_mode (str, optional): Image mode.

    Returns:
        list[Image.Image]: List containing the grid image.

    Example:
        ```py
        from embodied_gen.utils.process_media import combine_images_to_grid
        grid = combine_images_to_grid(["img1.png", "img2.png"])
        grid[0].save("grid.png")
        ```
    """
    n_images = len(images)
    if n_images == 0:
        return []
    if n_images == 1:
        return images

    if cat_row_col is None:
        n_col = math.ceil(math.sqrt(n_images))
        n_row = math.ceil(n_images / n_col)
    else:
        n_row, n_col = cat_row_col

    images = [
        Image.open(p).convert(image_mode) if isinstance(p, str) else p
        for p in images
    ]
    images = [img.resize(target_wh) for img in images]

    grid_w, grid_h = n_col * target_wh[0], n_row * target_wh[1]
    grid = Image.new(image_mode, (grid_w, grid_h), (0, 0, 0))

    for idx, img in enumerate(images):
        row, col = divmod(idx, n_col)
        grid.paste(img, (col * target_wh[0], row * target_wh[1]))

    return [grid]


def alpha_blend_rgba(
    fg_image: Union[str, Image.Image, np.ndarray],
    bg_image: Union[str, Image.Image, np.ndarray],
) -> Image.Image:
    """Alpha blends a foreground RGBA image over a background RGBA image.

    Args:
        fg_image: Foreground image (str, PIL Image, or ndarray).
        bg_image: Background image (str, PIL Image, or ndarray).

    Returns:
        Image.Image: Alpha-blended RGBA image.

    Example:
        ```py
        from embodied_gen.utils.process_media import alpha_blend_rgba
        result = alpha_blend_rgba("fg.png", "bg.png")
        result.save("blended.png")
        ```
    """
    if isinstance(fg_image, str):
        fg_image = Image.open(fg_image)
    elif isinstance(fg_image, np.ndarray):
        fg_image = Image.fromarray(fg_image)

    if isinstance(bg_image, str):
        bg_image = Image.open(bg_image)
    elif isinstance(bg_image, np.ndarray):
        bg_image = Image.fromarray(bg_image)

    if fg_image.size != bg_image.size:
        raise ValueError(
            f"Image sizes not match {fg_image.size} v.s. {bg_image.size}."
        )

    fg = fg_image.convert("RGBA")
    bg = bg_image.convert("RGBA")

    return Image.alpha_composite(bg, fg)


def check_object_edge_truncated(
    mask: np.ndarray, edge_threshold: int = 5
) -> bool:
    """Checks if a binary object mask is truncated at the image edges.

    Args:
        mask (np.ndarray): 2D binary mask.
        edge_threshold (int, optional): Edge pixel threshold.

    Returns:
        bool: True if object is fully enclosed, False if truncated.
    """
    top = mask[:edge_threshold, :].any()
    bottom = mask[-edge_threshold:, :].any()
    left = mask[:, :edge_threshold].any()
    right = mask[:, -edge_threshold:].any()

    return not (top or bottom or left or right)


def vcat_pil_images(
    images: list[Image.Image], image_mode: str = "RGB"
) -> Image.Image:
    """Vertically concatenates a list of PIL images.

    Args:
        images (list[Image.Image]): List of images.
        image_mode (str, optional): Image mode.

    Returns:
        Image.Image: Vertically concatenated image.

    Example:
        ```py
        from embodied_gen.utils.process_media import vcat_pil_images
        img = vcat_pil_images([Image.open("a.png"), Image.open("b.png")])
        img.save("vcat.png")
        ```
    """
    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights)
    max_width = max(widths)
    new_image = Image.new(image_mode, (max_width, total_height))
    y_offset = 0
    for image in images:
        new_image.paste(image, (0, y_offset))
        y_offset += image.size[1]

    return new_image


if __name__ == "__main__":
    image_paths = [
        "outputs/layouts_sim/task_0000/images/pen.png",
        "outputs/layouts_sim/task_0000/images/notebook.png",
        "outputs/layouts_sim/task_0000/images/mug.png",
        "outputs/layouts_sim/task_0000/images/lamp.png",
        "outputs/layouts_sim2/task_0014/images/cloth.png",  # TODO
    ]
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = image[..., -1]
        flag = check_object_edge_truncated(mask)
        print(flag, image_path)
