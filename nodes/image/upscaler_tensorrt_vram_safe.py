"""VRAM-conscious TensorRT upscaling.

The upstream TensorRT upscaler sends a complete frame to a dynamic engine.
That is fast when it fits, but TensorRT can reserve a very large activation
workspace before it ever produces the first pixel.  This node keeps the model
on the GPU while running fixed-size overlapping tiles and immediately moves
completed tiles to CPU memory.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar
import comfy.model_management as mm


MIN_TILE = 256
MAX_TILE = 1280
DEFAULT_TILE = 512
TILE_CHOICES = ["auto", "256", "384", "512", "640", "768", "1024"]


def _model_scale(model_name):
    """Read the installed TensorRT node's model config without importing it."""
    try:
        import folder_paths

        base_path = Path(getattr(folder_paths, "base_path", Path(folder_paths.models_dir).parent))
        config_path = (
            base_path
            / "custom_nodes"
            / "ComfyUI-Upscaler-Tensorrt"
            / "load_upscaler_config.json"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        for model in config.get("models", []):
            if model.get("name") == model_name:
                return int(model.get("scale", 4))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass

    match = re.match(r"^(\d+(?:\.\d+)?)x(?:[-_]|$)", str(model_name or ""))
    if match:
        return max(1, int(float(match.group(1))))
    return 4


def _final_resolution(width, height, resize_to, scale, resize_width, resize_height):
    if resize_to == "custom":
        return max(1, int(resize_width)), max(1, int(resize_height))
    if resize_to == "none":
        return width * scale, height * scale

    presets = {
        "HD": (1280, 720),
        "FHD": (1920, 1080),
        "2k": (2560, 1440),
        "4k": (3840, 2160),
    }
    if resize_to in presets:
        target_width, target_height = presets[resize_to]
        if height > width:
            target_width, target_height = target_height, target_width
        return target_width, target_height

    factor = float(str(resize_to).removesuffix("x"))
    return max(1, int(width * factor)), max(1, int(height * factor))


def _tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _is_oom(error):
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cudaerror 2",
            "error code 2",
            "could not allocate memory",
            "memory allocation",
        )
    )


def _release_vram(model, unload_models=False):
    """Release TensorRT's context and make ComfyUI's allocator reusable."""
    try:
        model.reset()
    except Exception:
        # A failed context creation can leave the third-party object partially
        # initialized.  The cache cleanup below is still useful in that case.
        pass
    if unload_models:
        try:
            mm.unload_all_models()
        except Exception:
            pass
    gc.collect()
    try:
        mm.soft_empty_cache(force=True)
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _initial_tile_size(choice):
    if choice != "auto":
        return max(MIN_TILE, min(MAX_TILE, int(choice)))
    if not torch.cuda.is_available():
        return DEFAULT_TILE
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / (1024 ** 3)
        if free_gib < 10:
            return 256
        if free_gib < 18:
            return 384
        if free_gib < 28:
            return 512
        return 640
    except RuntimeError:
        return DEFAULT_TILE


def _retry_tile_sizes(initial):
    sizes = []
    for size in (initial, 1024, 768, 640, 512, 384, 256):
        if MIN_TILE <= size <= MAX_TILE and size <= initial and size not in sizes:
            sizes.append(size)
    return sizes


class UpscalerTensorrtVramSafe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "upscaler_trt_model": ("UPSCALER_TRT_MODEL",),
                "resize_to": (
                    [
                        "none", "custom", "HD", "FHD", "2k", "4k",
                        "1x", "1.5x", "2x", "2.5x", "3x", "3.5x",
                        "4x", "5x", "6x", "7x", "8x", "9x", "10x",
                    ],
                    {"default": "none"},
                ),
                "resize_width": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "resize_height": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "tile_size": (TILE_CHOICES, {"default": "auto"}),
                "tile_overlap": (
                    "INT",
                    {"default": 32, "min": 0, "max": 256, "step": 8},
                ),
                "unload_models_on_oom": (
                    ["on_failure", "before_run", "never"],
                    {"default": "on_failure"},
                ),
            }
        }

    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    DESCRIPTION = (
        "TensorRT upscaling with fixed overlapping tiles, CPU output staging, "
        "and automatic smaller-tile retry after a CUDA/TensorRT OOM."
    )

    def _run_tiled(
        self,
        images_bchw,
        model,
        scale,
        tile_size,
        tile_overlap,
        final_width,
        final_height,
        pbar,
    ):
        batch, channels, height, width = images_bchw.shape
        raw_width = width * scale
        raw_height = height * scale
        output = torch.empty(
            (batch, channels, final_height, final_width),
            dtype=torch.float32,
            device="cpu",
        )
        overlap = min(max(0, int(tile_overlap)), tile_size // 2 - 1)
        y_starts = _tile_starts(height, tile_size, overlap)
        x_starts = _tile_starts(width, tile_size, overlap)

        model.reset()
        model.activate()
        model.allocate_buffers(
            shape_dict={
                "input": {"shape": (1, channels, tile_size, tile_size)},
                "output": {
                    "shape": (1, channels, tile_size * scale, tile_size * scale)
                },
            }
        )
        stream = torch.cuda.current_stream().cuda_stream

        for frame_index in range(batch):
            frame = images_bchw[frame_index : frame_index + 1]
            accumulated = torch.zeros(
                (channels, raw_height, raw_width), dtype=torch.float32
            )
            weights = torch.zeros((1, raw_height, raw_width), dtype=torch.float32)

            for top in y_starts:
                for left in x_starts:
                    bottom = min(top + tile_size, height)
                    right = min(left + tile_size, width)
                    patch_height = bottom - top
                    patch_width = right - left
                    patch = frame[:, :, top:bottom, left:right]
                    if patch_height != tile_size or patch_width != tile_size:
                        patch = F.pad(
                            patch,
                            (0, tile_size - patch_width, 0, tile_size - patch_height),
                            mode="replicate",
                        )

                    result = model.infer({"input": patch}, stream)["output"]
                    tile = result[0, :, : patch_height * scale, : patch_width * scale]
                    tile = tile.to(device="cpu", dtype=torch.float32)
                    tile_height, tile_width = tile.shape[-2:]

                    window = torch.ones(
                        (1, tile_height, tile_width), dtype=torch.float32
                    )
                    edge_y = min(overlap * scale, tile_height // 2)
                    edge_x = min(overlap * scale, tile_width // 2)
                    if top > 0 and edge_y:
                        window[:, :edge_y] *= torch.linspace(
                            0.0, 1.0, edge_y
                        ).reshape(1, edge_y, 1)
                    if bottom < height and edge_y:
                        window[:, -edge_y:] *= torch.linspace(
                            1.0, 0.0, edge_y
                        ).reshape(1, edge_y, 1)
                    if left > 0 and edge_x:
                        window[:, :, :edge_x] *= torch.linspace(
                            0.0, 1.0, edge_x
                        ).reshape(1, 1, edge_x)
                    if right < width and edge_x:
                        window[:, :, -edge_x:] *= torch.linspace(
                            1.0, 0.0, edge_x
                        ).reshape(1, 1, edge_x)

                    y0, y1 = top * scale, bottom * scale
                    x0, x1 = left * scale, right * scale
                    accumulated[:, y0:y1, x0:x1] += tile * window
                    weights[:, y0:y1, x0:x1] += window

            frame_output = accumulated / weights.clamp_min(1e-6)
            if (raw_width, raw_height) != (final_width, final_height):
                frame_output = F.interpolate(
                    frame_output.unsqueeze(0),
                    size=(final_height, final_width),
                    mode="bicubic",
                    antialias=True,
                )[0]
            output[frame_index] = frame_output
            pbar.update(1)

        return output

    def upscale(
        self,
        images,
        upscaler_trt_model,
        resize_to,
        resize_width,
        resize_height,
        tile_size,
        tile_overlap,
        unload_models_on_oom,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("VRAM-safe TensorRT upscaling requires CUDA.")

        images_bchw = images.permute(0, 3, 1, 2).contiguous()
        batch, channels, height, width = images_bchw.shape
        if channels != 3:
            raise ValueError("TensorRT upscaling expects RGB images with 3 channels.")

        model_name = getattr(upscaler_trt_model, "model_name", None)
        scale = _model_scale(model_name)
        final_width, final_height = _final_resolution(
            width, height, resize_to, scale, resize_width, resize_height
        )
        initial_tile = _initial_tile_size(tile_size)
        candidates = _retry_tile_sizes(initial_tile)
        if unload_models_on_oom == "before_run":
            _release_vram(upscaler_trt_model, unload_models=True)

        pbar = ProgressBar(batch)
        last_error = None
        try:
            for index, candidate in enumerate(candidates):
                try:
                    return (
                        self._run_tiled(
                            images_bchw,
                            upscaler_trt_model,
                            scale,
                            candidate,
                            tile_overlap,
                            final_width,
                            final_height,
                            pbar,
                        ).permute(0, 2, 3, 1),
                    )
                except Exception as error:
                    last_error = error
                    if not _is_oom(error) or index == len(candidates) - 1:
                        raise
                    _release_vram(
                        upscaler_trt_model,
                        unload_models=unload_models_on_oom != "never",
                    )
        finally:
            _release_vram(upscaler_trt_model)

        raise RuntimeError("TensorRT upscaling failed after all tile sizes.") from last_error
