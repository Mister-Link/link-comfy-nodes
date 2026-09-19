from __future__ import annotations

import json
import os

import folder_paths

from .load_folder import (
    BIGMAX,
    _build_preview_images,
    _is_changed_load_folder,
    _load_images,
    _resolve_load_folder_directory,
    _validate_load_folder,
)


class LoadFolderAnchorNode:
    """Load a frame folder and attach one fixed anchor coordinate to the batch.

    The anchor is deliberately a single point. It is not a per-frame tracking
    result: the same coordinate is used for every frame in the animation.
    """

    CATEGORY = "image/io"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "anchor", "frame_count")
    FUNCTION = "load_images"

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        directories = [
            item
            for item in os.listdir(input_dir)
            if not os.path.isfile(os.path.join(input_dir, item))
            and item != "clipspace"
        ]
        directories.sort(key=str.lower)
        return {
            "required": {
                "directory": (directories,),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                "anchor_x": ("INT", {"default": -1, "min": -1, "max": BIGMAX, "step": 1}),
                "anchor_y": ("INT", {"default": -1, "min": -1, "max": BIGMAX, "step": 1}),
            },
        }

    @staticmethod
    def _resolve_anchor(anchor_x: int, anchor_y: int, width: int, height: int) -> dict:
        # -1 means "use the canvas center" until the user clicks the preview.
        return {
            "x": int(width // 2 if anchor_x < 0 else max(0, min(width - 1, anchor_x))),
            "y": int(height // 2 if anchor_y < 0 else max(0, min(height - 1, anchor_y))),
        }

    def load_images(
        self,
        directory: str,
        image_load_cap: int = 0,
        skip_first_images: int = 0,
        select_every_nth: int = 1,
        anchor_x: int = -1,
        anchor_y: int = -1,
    ):
        directory = _resolve_load_folder_directory(directory)
        images, masks, frame_count = _load_images(
            directory,
            image_load_cap=image_load_cap,
            skip_first_images=skip_first_images,
            select_every_nth=select_every_nth,
        )
        height, width = images.shape[1:3]
        anchor = self._resolve_anchor(anchor_x, anchor_y, width, height)
        metadata = {
            "format": "link-comfy-nodes/anchor-v1",
            "sourceSize": {"w": int(width), "h": int(height)},
            "anchor": anchor,
        }
        preview_images = _build_preview_images(
            directory,
            image_load_cap=image_load_cap,
            skip_first_images=skip_first_images,
            select_every_nth=select_every_nth,
        )
        return {
            "ui": {"fast_images": preview_images},
            "result": (images, masks, json.dumps(metadata), frame_count),
        }

    @classmethod
    def IS_CHANGED(cls, directory: str, **kwargs):
        directory = _resolve_load_folder_directory(directory)
        # Include the one static anchor in cache invalidation so moving it
        # reruns downstream stabilization even when the files are unchanged.
        file_key = _is_changed_load_folder(directory, **kwargs)
        return f"{file_key}:{kwargs.get('anchor_x', -1)}:{kwargs.get('anchor_y', -1)}"

    @classmethod
    def VALIDATE_INPUTS(cls, directory: str, **kwargs):
        directory = _resolve_load_folder_directory(directory)
        return _validate_load_folder(directory)
