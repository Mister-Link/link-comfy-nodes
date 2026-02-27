from __future__ import annotations

import os
import re

import numpy as np
import torch
from PIL import Image

import folder_paths  # type: ignore[import-untyped]
from comfy_execution.utils import get_executing_context

_BATCH_IMAGE_SAVE_CACHE: dict[tuple[str, str], tuple[str, str]] = {}


class BatchImageSave:
    CATEGORY: str = "image"
    RETURN_TYPES: tuple[str, ...] = ("STRING", "STRING")
    RETURN_NAMES: tuple[str, ...] = ("folder_path", "file_names")
    FUNCTION: str = "save_images"
    OUTPUT_NODE: bool = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "path": (
                    "STRING",
                    {"default": "batch", "multiline": False},
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "frame", "multiline": False},
                ),
                "delimiter": (
                    "STRING",
                    {"default": "_", "multiline": False},
                ),
                "extension": (
                    ["png", "jpg", "jpeg", "webp"],
                    {"default": "png"},
                ),
                "link_to_input": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def save_images(
        self,
        images: torch.Tensor,
        path: str,
        filename_prefix: str,
        delimiter: str,
        extension: str,
        link_to_input: bool,
        unique_id: str | None = None,
    ):
        images_np = images.cpu().numpy()

        if images_np.ndim != 4:
            raise ValueError(
                f"Expected images with shape (N, H, W, C), got shape {images_np.shape}"
            )

        num_images = images_np.shape[0]
        saved_files = []

        output_dir = folder_paths.get_output_directory()

        ext = extension.lower()
        if ext == "jpg":
            ext = "jpeg"

        format_map = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
        pil_format = format_map.get(ext, "PNG")

        resolved_path = path
        full_dir = output_dir
        if path:
            executing_context = get_executing_context()
            list_index = executing_context.list_index if executing_context else None
            cache_owner = unique_id
            if cache_owner is None and executing_context is not None:
                cache_owner = executing_context.node_id
            cache_key = (cache_owner or "unknown", path)
            use_cached_folder = (
                list_index is not None
                and list_index > 0
                and cache_key in _BATCH_IMAGE_SAVE_CACHE
            )

            if use_cached_folder:
                resolved_path, full_dir = _BATCH_IMAGE_SAVE_CACHE[cache_key]
                os.makedirs(full_dir, exist_ok=True)
            else:

                def format_path(template: str, idx: int) -> str:
                    if "{index" in template:
                        return template.format_map({"index": idx})
                    return template.format(idx)

                uses_template = False
                if "{" in path and "}" in path:
                    try:
                        test_candidate = format_path(path, 1)
                        uses_template = test_candidate != path
                    except (ValueError, KeyError, IndexError):
                        uses_template = False

                if uses_template:
                    index = 1
                    while True:
                        try:
                            candidate = format_path(path, index)
                        except (ValueError, KeyError, IndexError):
                            uses_template = False
                            break
                        candidate_dir = os.path.join(output_dir, candidate)
                        if not os.path.exists(candidate_dir):
                            resolved_path = candidate
                            full_dir = candidate_dir
                            break
                        index += 1

                if not uses_template:
                    candidate_dir = os.path.join(output_dir, path)
                    if os.path.exists(candidate_dir):
                        suffix = 1
                        while True:
                            candidate = f"{path}_{suffix}"
                            candidate_dir = os.path.join(output_dir, candidate)
                            if not os.path.exists(candidate_dir):
                                resolved_path = candidate
                                full_dir = candidate_dir
                                break
                            suffix += 1
                    else:
                        resolved_path = path
                        full_dir = candidate_dir

                os.makedirs(full_dir, exist_ok=True)
                if list_index is not None:
                    _BATCH_IMAGE_SAVE_CACHE[cache_key] = (resolved_path, full_dir)

        if link_to_input and resolved_path:
            input_dir = folder_paths.get_input_directory()
            link_path = os.path.join(input_dir, f"[O]{resolved_path}")
            link_parent = os.path.dirname(link_path)
            os.makedirs(link_parent, exist_ok=True)

            if os.path.lexists(link_path):
                if os.path.islink(link_path):
                    existing_target = os.readlink(link_path)
                    if not os.path.isabs(existing_target):
                        existing_target = os.path.join(
                            os.path.dirname(link_path),
                            existing_target,
                        )
                    if os.path.abspath(existing_target) == os.path.abspath(full_dir):
                        link_path = None
                if link_path:
                    suffix = 1
                    base = link_path
                    while os.path.lexists(link_path):
                        link_path = f"{base}__link_{suffix}"
                        suffix += 1

            if link_path:
                os.symlink(full_dir, link_path)

        existing_max = 0
        pattern = re.compile(
            rf"^{re.escape(filename_prefix)}{re.escape(delimiter)}(\d+)\.{re.escape(ext)}$"
        )
        for name in os.listdir(full_dir):
            match = pattern.match(name)
            if match:
                existing_max = max(existing_max, int(match.group(1)))

        start_index = existing_max + 1

        digits = max(2, len(str(start_index + num_images - 1)))

        for i in range(num_images):
            index = start_index + i
            filename = f"{filename_prefix}{delimiter}{index:0{digits}d}.{ext}"

            relative_path = os.path.join(path, filename) if path else filename
            full_path = os.path.join(full_dir, filename)

            frame = images_np[i]
            frame = np.clip(frame, 0.0, 1.0)

            if frame.shape[-1] == 1:
                frame_255 = (frame[:, :, 0] * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_255, mode="L")
            elif frame.shape[-1] == 3:
                frame_255 = (frame * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_255, mode="RGB")
            elif frame.shape[-1] == 4:
                frame_255 = (frame * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_255, mode="RGBA")
                if pil_format == "JPEG":
                    pil_img = pil_img.convert("RGB")
            else:
                raise ValueError(f"Unsupported number of channels: {frame.shape[-1]}")

            pil_img.save(full_path, format=pil_format)
            saved_files.append(relative_path)

        folder_path = resolved_path if resolved_path else ""
        file_names = "\n".join([os.path.basename(f) for f in saved_files])

        return {
            "ui": {
                "text": [f"Saved {num_images} images to {resolved_path or 'output'}"]
            },
            "result": (folder_path, file_names),
        }
