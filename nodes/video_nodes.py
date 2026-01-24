from __future__ import annotations

import base64
import json
import os
import re
import zipfile
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol, TypedDict, cast

import cv2
import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.utils
import folder_paths  # type: ignore[import-untyped]
from comfy_execution.utils import get_executing_context
from node_helpers import conditioning_set_values  # type: ignore[import-not-found]

from ..utils import parse_hex_color

try:
    from aiohttp import web

    from server import PromptServer  # type: ignore[import-not-found]
except Exception:
    PromptServer = None
    web = None

VIDEO_EXTENSIONS = ["webm", "mp4", "mkv", "gif", "mov"]
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"]
BIGMAX = 2**53 - 1
DIMMAX = 8192
DEFAULT_PREVIEW_FRAME_LIMIT = 120
PREVIEW_CACHE_MAX_ITEMS = 10  # Cache more previews with optimized WebP compression
PREVIEW_CACHE_MAX_BYTES = 30_000_000  # Allow larger cache since WebP is very efficient
_BATCH_IMAGE_SAVE_CACHE: dict[tuple[str, str], tuple[str, str]] = {}


def _log(message: str) -> None:
    _ = message  # Suppress unused parameter warning


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str, bytes, bytearray)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


class MaskBbox(TypedDict, total=False):
    x: int
    y: int
    width: int
    height: int


class MaskKeyframe(TypedDict, total=False):
    type: str
    bbox: MaskBbox
    mask_data: str
    mask_width: int
    mask_height: int


class _RelUrl(Protocol):
    query: Mapping[str, str]


class _Request(Protocol):
    rel_url: _RelUrl

    async def json(self) -> dict[str, object]: ...


class FrameLoadResult(TypedDict):
    frames: list[np.ndarray]
    selected_indices: list[int]
    target_width: int
    target_height: int
    original_fps: int
    effective_fps: float
    total_frames: int
    frame_step: int
    combined_step: int


@dataclass
class MaskRegion:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, object] | None) -> "MaskRegion | None":
        if not payload:
            return None
        try:
            return cls(
                x=_coerce_int(payload.get("x", 0)),
                y=_coerce_int(payload.get("y", 0)),
                width=_coerce_int(payload.get("width", 0)),
                height=_coerce_int(payload.get("height", 0)),
            )
        except Exception:
            return None

    def clamp(self, max_width: int, max_height: int) -> "MaskRegion":
        x = max(0, min(self.x, max_width - 1))
        y = max(0, min(self.y, max_height - 1))
        width = max(0, min(self.width, max_width - x))
        height = max(0, min(self.height, max_height - y))
        return MaskRegion(x=x, y=y, width=width, height=height)


_mask_regions: dict[str, MaskRegion] = {}
_mask_regions_by_video: dict[str, MaskRegion] = {}
_mask_versions: dict[str, int] = {}
_mask_keyframes: dict[str, dict[str, MaskKeyframe]] = {}
_preview_cache: OrderedDict[str, dict[str, object]] = OrderedDict()

# Persistence file path
_KEYFRAMES_CACHE_FILE = os.path.join(
    folder_paths.get_output_directory(), "video_mask_keyframes.json"
)


def _save_keyframes_to_disk():
    """Save keyframes to disk for persistence."""
    try:
        with open(_KEYFRAMES_CACHE_FILE, "w") as f:
            json.dump(_mask_keyframes, f)
    except Exception as e:
        _log(f"Failed to save keyframes to disk: {e}")


def _load_keyframes_from_disk():
    """Load keyframes from disk."""
    global _mask_keyframes
    try:
        if os.path.exists(_KEYFRAMES_CACHE_FILE):
            with open(_KEYFRAMES_CACHE_FILE, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    _mask_keyframes = cast(dict[str, dict[str, MaskKeyframe]], loaded)
                else:
                    _mask_keyframes = {}
            _log(f"Loaded keyframes from disk: {len(_mask_keyframes)} nodes")
    except Exception as e:
        _log(f"Failed to load keyframes from disk: {e}")


def _clear_stale_masks():
    """Clear mask regions to prevent stale data."""
    _mask_regions.clear()
    _mask_regions_by_video.clear()
    _mask_versions.clear()
    _mask_keyframes.clear()
    _save_keyframes_to_disk()
    _log("Cleared all mask regions and keyframes")
    _preview_cache.clear()


def _increment_mask_version(node_id: str) -> int:
    """Track mask updates to invalidate execution cache."""
    if not node_id:
        return 0
    node_id = str(node_id)
    _mask_versions[node_id] = _mask_versions.get(node_id, 0) + 1
    _log(f"Mask version for node {node_id} -> {_mask_versions[node_id]}")
    _preview_cache.clear()  # Avoid serving stale cached previews
    return _mask_versions[node_id]


def _get_mask_version(node_id: str | None) -> int:
    if not node_id:
        return 0
    return _mask_versions.get(str(node_id), 0)


def _preview_cache_key(
    video_path: str,
    video_mtime: float,
    framerate: int,
    custom_width: int,
    custom_height: int,
    frame_load_cap: int,
    skip_first_frames: int,
    select_every_nth: int,
    max_preview_frames: int,
    skip_mask: bool,
    mask_version: int,
) -> str:
    """Create a stable cache key for preview responses."""
    key_tuple = (
        video_path,
        round(video_mtime, 3),
        framerate,
        custom_width,
        custom_height,
        frame_load_cap,
        skip_first_frames,
        select_every_nth,
        max_preview_frames,
        skip_mask,
        mask_version,
    )
    return json.dumps(key_tuple, separators=(",", ":"))


def _maybe_cache_preview(key: str, payload: dict[str, object]) -> None:
    """Cache preview payload when it is reasonably small."""
    try:
        frames = payload.get("frames")
        if not isinstance(frames, list):
            total_bytes = PREVIEW_CACHE_MAX_BYTES + 1
        else:
            total_bytes = sum(
                len(frame.get("data", ""))
                for frame in frames
                if isinstance(frame, dict)
            )
    except Exception:
        total_bytes = PREVIEW_CACHE_MAX_BYTES + 1

    if total_bytes > PREVIEW_CACHE_MAX_BYTES:
        _log(
            f"Skipping preview cache (payload too large: {total_bytes} bytes, limit {PREVIEW_CACHE_MAX_BYTES})"
        )
        return

    _preview_cache[key] = payload
    _preview_cache.move_to_end(key)

    while len(_preview_cache) > PREVIEW_CACHE_MAX_ITEMS:
        _ = _preview_cache.popitem(last=False)


def _generate_masks_from_keyframes(
    node_id: str | None, frame_count: int, width: int, height: int
) -> np.ndarray:
    """Generate masks for all frames based on keyframe data.

    Keyframe propagation logic:
    - Frames before the first keyframe have blank masks (all zeros)
    - Frames at or after a keyframe use that keyframe's mask until the next keyframe
    - Each keyframe's mask applies forward until another keyframe is encountered
    - Hybrid keyframes combine both bbox and painted masks (union)
    """
    if not node_id or str(node_id) not in _mask_keyframes:
        # No keyframes, return all-zero masks (blank)
        return np.zeros((frame_count, height, width), dtype=np.float32)

    keyframes = _mask_keyframes[str(node_id)]
    if not keyframes:
        return np.zeros((frame_count, height, width), dtype=np.float32)

    # Sort keyframes by frame index
    sorted_keyframes = sorted(keyframes.items(), key=lambda x: int(x[0]))
    keyframe_indices = [int(kf[0]) for kf in sorted_keyframes]

    _log(
        f"Generating masks for node {node_id}: {len(keyframes)} keyframes, {frame_count} frames"
    )
    for kf_idx, kf_data in sorted_keyframes:
        _log(
            f"  Keyframe {kf_idx}: type={kf_data.get('type')}, has_bbox={bool(kf_data.get('bbox'))}, has_mask_data={bool(kf_data.get('mask_data'))}"
        )

    masks = np.zeros((frame_count, height, width), dtype=np.float32)

    # Process each frame
    for frame_idx in range(frame_count):
        # Find the active keyframe for this frame:
        # - If before first keyframe, leave as blank (zero mask)
        # - Otherwise, interpolate between keyframes or use the most recent one

        if frame_idx < keyframe_indices[0]:
            # Before the first keyframe - leave mask as blank (zeros)
            continue

        # Find surrounding keyframes for interpolation
        prev_kf_idx = None
        next_kf_idx = None
        prev_kf_data = None
        next_kf_data = None

        for kf_idx in keyframe_indices:
            if kf_idx <= frame_idx:
                prev_kf_idx = kf_idx
                prev_kf_data = keyframes[str(kf_idx)]
            if kf_idx > frame_idx and next_kf_idx is None:
                next_kf_idx = kf_idx
                next_kf_data = keyframes[str(kf_idx)]
                break

        if prev_kf_data is None:
            # No previous keyframe, leave as blank
            continue

        mask_type = prev_kf_data.get("type", "bbox")

        # Skip empty keyframes
        if mask_type == "empty":
            continue

        # Handle bbox mask (including hybrid)
        if mask_type in ("bbox", "hybrid") and prev_kf_data.get("bbox"):
            # Interpolate bbox position between keyframes
            prev_bbox = prev_kf_data.get("bbox", {})

            if (
                next_kf_data
                and next_kf_data.get("bbox")
                and next_kf_idx is not None
                and prev_kf_idx is not None
            ):
                # Interpolate between two bbox keyframes
                next_bbox = next_kf_data.get("bbox", {})

                # Calculate interpolation factor (0.0 at prev, 1.0 at next)
                total_frames = next_kf_idx - prev_kf_idx
                if total_frames > 0:
                    t = (frame_idx - prev_kf_idx) / total_frames
                else:
                    t = 0.0

                # Linear interpolation for all bbox properties
                x = int(prev_bbox.get("x", 0) * (1 - t) + next_bbox.get("x", 0) * t)
                y = int(prev_bbox.get("y", 0) * (1 - t) + next_bbox.get("y", 0) * t)
                w = int(
                    prev_bbox.get("width", width) * (1 - t)
                    + next_bbox.get("width", width) * t
                )
                h = int(
                    prev_bbox.get("height", height) * (1 - t)
                    + next_bbox.get("height", height) * t
                )
            else:
                # No next keyframe or different type, just use previous bbox
                x = int(prev_bbox.get("x", 0))
                y = int(prev_bbox.get("y", 0))
                w = int(prev_bbox.get("width", width))
                h = int(prev_bbox.get("height", height))

            # Clamp to bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(0, min(w, width - x))
            h = max(0, min(h, height - y))

            masks[frame_idx, y : y + h, x : x + w] = 1.0

        # Handle painted mask (including hybrid)
        if mask_type in ("painted", "hybrid") and prev_kf_data.get("mask_data"):
            # Decode painted mask from base64
            mask_data = prev_kf_data.get("mask_data", "")
            if mask_data:
                try:
                    _log(
                        f"Processing painted mask for frame {frame_idx}, type={mask_type}, data_len={len(mask_data)}"
                    )
                    decoded = base64.b64decode(mask_data)
                    mask_array = np.frombuffer(decoded, dtype=np.float32)

                    # Get stored mask dimensions (from when it was painted)
                    stored_width = prev_kf_data.get("mask_width", width)
                    stored_height = prev_kf_data.get("mask_height", height)

                    _log(
                        f"Mask dimensions: stored=({stored_height}, {stored_width}), target=({height}, {width})"
                    )

                    # Reshape to original dimensions
                    expected_size = stored_height * stored_width
                    if len(mask_array) != expected_size:
                        _log(
                            f"WARNING: Mask size mismatch. Expected {expected_size}, got {len(mask_array)}"
                        )
                        # Try to use the mask data length to infer dimensions
                        stored_height = int(np.sqrt(len(mask_array)))
                        stored_width = len(mask_array) // stored_height

                    mask_array = mask_array.reshape((stored_height, stored_width))

                    # Resize mask if dimensions don't match target
                    if stored_width != width or stored_height != height:
                        _log(
                            f"Resizing mask from ({stored_height}, {stored_width}) to ({height}, {width})"
                        )
                        # Use OpenCV to resize the mask
                        mask_array = cv2.resize(
                            mask_array,
                            (width, height),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    # Union with existing mask (take maximum)
                    masks[frame_idx] = np.maximum(masks[frame_idx], mask_array)
                    _log(
                        f"Applied painted mask, non-zero pixels: {np.count_nonzero(mask_array > 0.5)}"
                    )
                except Exception as e:
                    _log(f"Failed to decode painted mask: {e}")
                    import traceback

                    _log(traceback.format_exc())

    return masks


def _calculate_target_size(
    width: int,
    height: int,
    custom_width: int,
    custom_height: int,
    downscale_ratio: int = 8,
) -> tuple[int, int]:
    if custom_width == 0 and custom_height == 0:
        target_width, target_height = width, height
    elif custom_height == 0:
        target_width = custom_width
        target_height = int(height * (custom_width / width))
    elif custom_width == 0:
        target_width = int(width * (custom_height / height))
        target_height = custom_height
    else:
        target_width, target_height = custom_width, custom_height

    target_width = int(target_width / downscale_ratio + 0.5) * downscale_ratio
    target_height = int(target_height / downscale_ratio + 0.5) * downscale_ratio
    return target_width, target_height


def _load_frames_from_folder(
    folder_path: str,
    framerate: int,
    custom_width: int,
    custom_height: int,
    frame_load_cap: int,
    skip_first_frames: int,
    select_every_nth: int,
    preview_max_frames: int | None = None,
) -> FrameLoadResult:
    """Load frames from a folder of images."""
    if not os.path.isdir(folder_path):
        raise ValueError(f"Not a directory: {folder_path}")

    # Get all image files
    image_files = []
    for fname in os.listdir(folder_path):
        if fname.split(".")[-1].lower() in IMAGE_EXTENSIONS:
            image_files.append(os.path.join(folder_path, fname))

    if not image_files:
        raise ValueError(f"No image files found in folder: {folder_path}")

    # Sort files by name
    image_files.sort()

    # Get dimensions from first image
    first_img = Image.open(image_files[0])
    width, height = first_img.size
    first_img.close()

    if width <= 0 or height <= 0:
        raise ValueError("Could not determine image dimensions")

    target_width, target_height = _calculate_target_size(
        width, height, custom_width, custom_height
    )

    frames_list: list[np.ndarray] = []
    selected_frame_indices: list[int] = []

    max_frames = frame_load_cap if frame_load_cap > 0 else None
    if preview_max_frames is not None and preview_max_frames > 0:
        max_frames = (
            min(max_frames, preview_max_frames)
            if max_frames is not None
            else preview_max_frames
        )

    # Calculate frame sampling based on framerate
    # For image folders, assume 18 fps (not 24 as previously)
    # If framerate is 0 or not specified, use all frames (frame_step = 1)
    # If framerate > 0, calculate which frames to select:
    # - We want to select frames at time intervals of 1/framerate seconds
    # - At the assumed fps, this means selecting every (assumed_fps / framerate) frames
    # Example: assumed_fps=18, framerate=3 -> frame_step = 18/3 = 6
    # So we select frames 0, 6, 12, 18, ... (at times 0, 1/3s, 2/3s, 1s, ...)
    assumed_fps = 18.0
    frame_step = (
        max(1, int(round(assumed_fps / framerate)))
        if framerate and framerate > 0
        else 1
    )
    combined_step = max(1, select_every_nth) * frame_step

    # Process images
    for frame_index, img_path in enumerate(image_files):
        if frame_index < skip_first_frames:
            continue

        relative_index = frame_index - skip_first_frames
        if relative_index % combined_step != 0:
            continue

        try:
            img = Image.open(img_path)
            # Preserve alpha channel if present, otherwise convert to RGB
            if img.mode == "RGBA":
                # Keep RGBA mode to preserve alpha channel
                pass
            elif img.mode != "RGB":
                img = img.convert("RGB")

            if target_width != width or target_height != height:
                img = img.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )

            frame = np.array(img, dtype=np.float32) / 255.0
            if frame.ndim == 2:
                frame = np.expand_dims(frame, axis=2)

            # Keep all channels for now, will separate RGB/alpha later
            frames_list.append(frame)
            selected_frame_indices.append(frame_index)
            img.close()

            if max_frames is not None and len(frames_list) >= max_frames:
                break
        except Exception as e:
            _log(f"Failed to load image {img_path}: {e}")
            continue

    if not frames_list:
        raise RuntimeError("No frames loaded from folder")

    # For image sequences, assume 24 fps
    # effective_fps is the target framerate if specified, otherwise assumed_fps
    effective_fps = framerate if framerate > 0 else assumed_fps

    return {
        "frames": frames_list,
        "selected_indices": selected_frame_indices,
        "target_width": target_width,
        "target_height": target_height,
        "original_fps": int(assumed_fps),  # Assumed FPS for image sequences
        "effective_fps": effective_fps,
        "total_frames": len(image_files),
        "frame_step": frame_step,
        "combined_step": combined_step,
    }


def _load_video_frames(
    video_path: str,
    framerate: int,
    custom_width: int,
    custom_height: int,
    frame_load_cap: int,
    skip_first_frames: int,
    select_every_nth: int,
    preview_max_frames: int | None = None,
) -> FrameLoadResult:
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        ret, frame = video_cap.read()
        if ret:
            height, width = frame.shape[:2]
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            video_cap.release()
            raise ValueError("Could not read video frame to determine dimensions")

    target_width, target_height = _calculate_target_size(
        width, height, custom_width, custom_height
    )

    # Calculate frame sampling based on framerate
    # If framerate is 0 or not specified, use original fps (frame_step = 1)
    # If framerate > 0, calculate which frames to select:
    # - We want to select frames at time intervals of 1/framerate seconds
    # - At the original fps, this means selecting every (original_fps / framerate) frames
    # Example: original_fps=18, framerate=3 -> frame_step = 18/3 = 6
    # So we select frames 0, 6, 12, 18, ... (at times 0, 1/3s, 2/3s, 1s, ...)
    frame_step = (
        max(1, int(round(original_fps / framerate)))
        if framerate and framerate > 0 and original_fps > 0
        else 1
    )
    combined_step = max(1, select_every_nth) * frame_step

    frames_list: list[np.ndarray] = []
    selected_frame_indices: list[int] = []
    frame_index = 0

    max_frames = frame_load_cap if frame_load_cap > 0 else None
    if preview_max_frames is not None and preview_max_frames > 0:
        max_frames = (
            min(max_frames, preview_max_frames)
            if max_frames is not None
            else preview_max_frames
        )

    try:
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break

            if frame_index < skip_first_frames:
                frame_index += 1
                continue

            relative_index = frame_index - skip_first_frames
            if relative_index % combined_step != 0:
                frame_index += 1
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_width != width or target_height != height:
                frame = cv2.resize(
                    frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            frame = np.array(frame, dtype=np.float32) / 255.0
            if frame.ndim == 2:
                frame = np.expand_dims(frame, axis=2)

            frames_list.append(frame)
            selected_frame_indices.append(frame_index)
            frame_index += 1

            if max_frames is not None and len(frames_list) >= max_frames:
                break
    finally:
        video_cap.release()

    if not frames_list:
        raise RuntimeError("No frames loaded from video")

    # effective_fps is the target framerate if specified, otherwise original fps
    effective_fps = framerate if framerate > 0 else original_fps

    return {
        "frames": frames_list,
        "selected_indices": selected_frame_indices,
        "target_width": target_width,
        "target_height": target_height,
        "original_fps": int(original_fps),
        "effective_fps": effective_fps,
        "total_frames": total_frames,
        "frame_step": frame_step,
        "combined_step": combined_step,
    }


class VideoMaskEditor:
    """Load a video, create bbox for regions, and expose preview endpoints."""

    CATEGORY: str = "Video/Masking"
    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "INT", "MASK", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("frames", "frame_count", "masks", "alpha_channel")
    FUNCTION: str = "load_video"
    OUTPUT_NODE: bool = False

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        items = []

        # Add video files
        for f in os.listdir(input_dir):
            full_path = os.path.join(input_dir, f)
            if (
                os.path.isfile(full_path)
                and f.split(".")[-1].lower() in VIDEO_EXTENSIONS
            ):
                items.append(f)
            # Add directories (potential image folders)
            elif os.path.isdir(full_path):
                items.append(f)

        return {
            "required": {
                "source": (sorted(items),),
                "framerate": (
                    "INT",
                    {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0},
                ),
                "frame_load_cap": (
                    "INT",
                    {"default": 0, "min": 0, "max": BIGMAX, "step": 1, "disable": 0},
                ),
                "skip_first_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": BIGMAX, "step": 1},
                ),
                "select_every_nth": (
                    "INT",
                    {"default": 1, "min": 1, "max": BIGMAX, "step": 1},
                ),
                "is_wan": ("BOOLEAN", {"default": False}),
                "bg_color": ("STRING", {"default": "#ffffff", "multiline": False}),
            },
            "hidden": {"force_size": "STRING", "unique_id": "UNIQUE_ID"},
        }

    def load_video(
        self,
        source: str,
        framerate: int,
        frame_load_cap: int,
        skip_first_frames: int,
        select_every_nth: int,
        is_wan: bool,
        bg_color: str,
        force_size: str = "",
        unique_id: str | None = None,
    ):
        _ = force_size  # Suppress unused parameter warning

        video_path = folder_paths.get_annotated_filepath(source)

        # WAN mode: enforce frame_load_cap follows 4n+1 formula (minimum 5)
        # Must consider available frames after skipping
        if is_wan:
            # Get total frames from source
            source_total_frames = 0
            if os.path.isdir(video_path):
                image_files = [
                    f
                    for f in os.listdir(video_path)
                    if f.split(".")[-1].lower() in IMAGE_EXTENSIONS
                ]
                source_total_frames = len(image_files)
                # For image folders, assume 18 fps
                assumed_fps = 18.0
                frame_step = (
                    max(1, int(round(assumed_fps / framerate)))
                    if framerate and framerate > 0
                    else 1
                )
            elif os.path.isfile(video_path):
                video_cap = None
                try:
                    video_cap = cv2.VideoCapture(video_path)
                    if video_cap.isOpened():
                        source_total_frames = int(
                            video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        )
                        original_fps = video_cap.get(cv2.CAP_PROP_FPS)
                        frame_step = (
                            max(1, int(round(original_fps / framerate)))
                            if framerate and framerate > 0 and original_fps > 0
                            else 1
                        )
                    else:
                        frame_step = 1
                finally:
                    if video_cap is not None:
                        video_cap.release()
            else:
                frame_step = 1

            # Calculate available frames after skipping and sampling
            combined_step = max(1, select_every_nth) * frame_step
            available_frames = max(0, source_total_frames - skip_first_frames)
            # Account for combined_step: we get frames at indices 0, combined_step, 2*combined_step, ...
            available_sampled = (available_frames + combined_step - 1) // combined_step

            # If frame_load_cap is 0, snap to largest WAN value that fits
            # If frame_load_cap > 0, snap to nearest WAN value that doesn't exceed available frames
            if frame_load_cap == 0:
                target_frames = available_sampled
            else:
                target_frames = min(frame_load_cap, available_sampled)

            # Snap to largest valid WAN value <= target_frames
            # Valid WAN values: 5, 9, 13, 17, 21, 25, ...
            n = max(1, (target_frames - 1) // 4)
            snapped_cap = 4 * n + 1
            if snapped_cap < 5:
                snapped_cap = 5

            if frame_load_cap != snapped_cap:
                _log(
                    f"WAN mode: Snapped frame_load_cap from {frame_load_cap} to {snapped_cap} "
                    f"(4n+1 formula, available after skip/sample: {available_sampled})"
                )
                frame_load_cap = snapped_cap

        # Detect if it's a directory (image folder) or file (video)
        if os.path.isdir(video_path):
            # Validate it's an image folder
            image_files = [
                f
                for f in os.listdir(video_path)
                if f.split(".")[-1].lower() in IMAGE_EXTENSIONS
            ]
            if not image_files:
                raise ValueError(f"Folder contains no image files: {source}")

            _log(f"Loading frames from image folder: {source}")
            processing_result = _load_frames_from_folder(
                video_path,
                framerate,
                0,  # custom_width (0 = use original)
                0,  # custom_height (0 = use original)
                frame_load_cap,
                skip_first_frames,
                select_every_nth,
            )
        elif os.path.isfile(video_path):
            # Validate it's a video file
            ext = source.split(".")[-1].lower()
            if ext not in VIDEO_EXTENSIONS:
                raise ValueError(f"Not a valid video file: {source}")

            _log(f"Loading frames from video file: {source}")
            processing_result = _load_video_frames(
                video_path,
                framerate,
                0,  # custom_width (0 = use original)
                0,  # custom_height (0 = use original)
                frame_load_cap,
                skip_first_frames,
                select_every_nth,
            )
        else:
            raise ValueError(
                f"Input is neither a valid video file nor a folder: {source}"
            )

        frames_list = processing_result["frames"]
        target_width = processing_result["target_width"]
        target_height = processing_result["target_height"]
        selected_frame_indices = processing_result["selected_indices"]
        effective_fps = processing_result["effective_fps"]
        original_fps = processing_result["original_fps"]
        total_frames = processing_result["total_frames"]
        frame_step = processing_result["frame_step"]
        combined_step = processing_result["combined_step"]

        _log(f"Loaded {len(frames_list)} frames")
        _log(
            f"Video: {original_fps} fps, {total_frames} frames (effective {effective_fps} fps)"
        )
        _log(
            f"Sampling every {combined_step} frame(s) (select_every_nth={select_every_nth}, frame_step={frame_step})"
        )

        frames_array = np.array(frames_list, dtype=np.float32)
        if frames_array.ndim == 3:
            frames_array = np.expand_dims(frames_array, axis=3)
        frames_array = np.clip(frames_array, 0.0, 1.0)

        # Separate RGB and alpha channels
        if frames_array.shape[-1] == 4:
            # Has alpha - split into RGB (3 channels) and alpha (1 channel)
            rgb_array = frames_array[:, :, :, :3]
            alpha_array = frames_array[:, :, :, 3]  # Shape: (N, H, W)
        else:
            # No alpha - use RGB as-is, create full opacity alpha
            rgb_array = frames_array
            if rgb_array.shape[-1] == 1:
                # Grayscale - convert to RGB
                rgb_array = np.repeat(rgb_array, 3, axis=3)
            # Create full alpha (all ones)
            alpha_array = np.ones(
                (frames_array.shape[0], frames_array.shape[1], frames_array.shape[2]),
                dtype=np.float32,
            )

        bg_rgb = np.array(parse_hex_color(bg_color, fallback=(255, 255, 255)))
        bg_rgb = (bg_rgb / 255.0).astype(np.float32)
        alpha_expanded = alpha_array[..., None]
        rgb_array = (rgb_array * alpha_expanded) + (bg_rgb * (1.0 - alpha_expanded))

        frames_tensor = torch.from_numpy(rgb_array)
        alpha_tensor = torch.from_numpy(alpha_array)

        _log(f"is_wan setting: {is_wan}")
        _log(f"unique_id: {unique_id}")
        _log(f"Available mask regions: {list(_mask_regions.keys())}")
        _log(f"_mask_regions dict id: {id(_mask_regions)}")

        region = None
        if unique_id and str(unique_id) in _mask_regions:
            region = _mask_regions[str(unique_id)]
            _log(f"Found region for unique_id {unique_id}: {region}")
        else:
            _log(f"No mask region found for unique_id: {unique_id}")

        bbox = {
            "x": 0,
            "y": 0,
            "width": target_width,
            "height": target_height,
        }

        if region:
            region = region.clamp(target_width, target_height)
            if region.width and region.height:
                bbox["x"] = region.x
                bbox["y"] = region.y
                bbox["width"] = region.width
                bbox["height"] = region.height
                _log(
                    f"Set bbox: x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}"
                )
            else:
                _log(
                    f"Region has zero width or height: w={region.width}, h={region.height}"
                )

        output_dir = folder_paths.get_output_directory()
        for idx in range(rgb_array.shape[0]):
            # Save RGB frames only (alpha saved separately)
            frame_rgb = (rgb_array[idx] * 255).astype(np.uint8)
            pil_img = Image.fromarray(frame_rgb)
            preview_name = f"vme_frame_{unique_id}_{idx:04d}.png"
            # Save with maximum quality (compress_level=1 is fast lossless, 0 is no compression)
            pil_img.save(
                os.path.join(output_dir, preview_name), compress_level=1, optimize=False
            )

        _log(
            f"Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}"
        )
        _ = json.dumps(selected_frame_indices)  # backward compatibility noop

        # Generate masks from keyframes
        masks_array = _generate_masks_from_keyframes(
            unique_id, frames_tensor.shape[0], target_width, target_height
        )

        masks_tensor = torch.from_numpy(masks_array)
        _log(f"Masks tensor shape: {masks_tensor.shape}, dtype: {masks_tensor.dtype}")
        _log(f"Alpha tensor shape: {alpha_tensor.shape}, dtype: {alpha_tensor.dtype}")

        return frames_tensor, frames_tensor.shape[0], masks_tensor, alpha_tensor

    @classmethod
    def IS_CHANGED(cls, source: str, **kwargs: object) -> str:
        video_path = folder_paths.get_annotated_filepath(source)
        unique_id = kwargs.get("unique_id")
        mask_version = _get_mask_version(
            str(unique_id) if unique_id is not None else None
        )
        video_mtime = os.path.getmtime(video_path) if os.path.exists(video_path) else -1

        key = (
            source,
            round(video_mtime, 3),
            kwargs.get("framerate", 0),
            kwargs.get("frame_load_cap", 0),
            kwargs.get("skip_first_frames", 0),
            kwargs.get("select_every_nth", 1),
            bool(kwargs.get("is_wan", False)),
            kwargs.get("bg_color", ""),
            kwargs.get("force_size", ""),
            str(unique_id),
            mask_version,
        )
        return json.dumps(key, separators=(",", ":"))

    @classmethod
    def VALIDATE_INPUTS(cls, source: str, **kwargs: object) -> str | bool:
        _ = kwargs
        if not folder_paths.exists_annotated_filepath(source):
            return f"Invalid video file: {source}"
        return True


def get_video_preview_frame(video_path: str, frame_number: int = 0):
    """Return a single frame from video for preview."""
    try:
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            return None

        video_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
        ret, frame = video_cap.read()
        video_cap.release()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)
        return None
    except Exception as exc:  # pragma: no cover - surfaced in UI
        _log(f"Error getting preview frame: {exc}")
        return None


def _register_preview_route():
    if PromptServer is None or web is None:
        return

    server = PromptServer
    aiohttp_web = web

    @server.instance.routes.get("/videomaskeditor/preview")
    async def video_mask_editor_preview(request: _Request):  # pylint: disable=unused-variable
        params = request.rel_url.query
        video_name = params.get("video")
        node_id = params.get("node_id")  # Added to get keyframes for this node

        if not video_name:
            return aiohttp_web.json_response(
                {"error": "Missing video parameter"}, status=400
            )

        if not folder_paths.exists_annotated_filepath(video_name):
            return aiohttp_web.json_response({"error": "Video not found"}, status=404)

        def _int_param(key: str, default: int, minimum: int | None = None) -> int:
            try:
                value = int(params.get(key, default))
            except (TypeError, ValueError):
                value = default
            if minimum is not None and value < minimum:
                value = minimum
            return value

        framerate = _int_param("framerate", 0, 0)
        custom_width = _int_param("custom_width", 0, 0)
        custom_height = _int_param("custom_height", 0, 0)
        frame_load_cap = _int_param("frame_load_cap", 0, 0)
        skip_first_frames = _int_param("skip_first_frames", 0, 0)
        select_every_nth = _int_param("select_every_nth", 1, 1)
        max_preview_frames = _int_param(
            "max_preview_frames", DEFAULT_PREVIEW_FRAME_LIMIT, 1
        )
        skip_mask = params.get("skip_mask", "false").lower() == "true"

        # Performance optimization: aggressive downscaling for faster preview loading
        # Scale down by 2x for mask editor, 1.5x for dialog to reduce file size and load time
        preview_downscale = 2.0 if skip_mask else 1.5
        if preview_downscale > 1:
            if custom_width > 0:
                custom_width = max(64, int(custom_width / preview_downscale))
            if custom_height > 0:
                custom_height = max(64, int(custom_height / preview_downscale))

        mask_version = _get_mask_version(node_id)
        cache_key = None

        processing_result: FrameLoadResult
        try:
            video_path = folder_paths.get_annotated_filepath(video_name)
            video_mtime = os.path.getmtime(video_path)

            cache_key = _preview_cache_key(
                video_path,
                video_mtime,
                framerate,
                custom_width,
                custom_height,
                frame_load_cap,
                skip_first_frames,
                select_every_nth,
                max_preview_frames,
                skip_mask,
                0 if skip_mask else mask_version,
            )

            if cache_key in _preview_cache:
                cached_payload = _preview_cache[cache_key]
                _preview_cache.move_to_end(cache_key)
                cached_frames = cached_payload.get("frames")
                cached_count = (
                    len(cached_frames) if isinstance(cached_frames, list) else 0
                )
                _log(f"Preview cache HIT for {video_name} ({cached_count} frames)")
                return aiohttp_web.json_response(cached_payload)

            # Detect if it's a directory or file
            if os.path.isdir(video_path):
                processing_result = _load_frames_from_folder(
                    video_path,
                    framerate,
                    custom_width,
                    custom_height,
                    frame_load_cap,
                    skip_first_frames,
                    select_every_nth,
                    preview_max_frames=max_preview_frames,
                )
            else:
                processing_result = _load_video_frames(
                    video_path,
                    framerate,
                    custom_width,
                    custom_height,
                    frame_load_cap,
                    skip_first_frames,
                    select_every_nth,
                    preview_max_frames=max_preview_frames,
                )
        except Exception as exc:  # pragma: no cover - surfaced in UI
            return aiohttp_web.json_response({"error": str(exc)}, status=400)

        # --- Generate masks and apply overlays ---
        masks_array = None
        if node_id and not skip_mask:
            try:
                masks_array = _generate_masks_from_keyframes(
                    node_id,
                    len(processing_result["frames"]),
                    processing_result["target_width"],
                    processing_result["target_height"],
                )
            except Exception as e:
                _log(f"Could not generate masks for preview: {e}")
        # --- End mask generation ---

        frames_payload: list[dict[str, object]] = []
        for idx, frame_data in enumerate(processing_result["frames"]):
            # Keep RGBA if present, otherwise use RGB
            has_alpha = frame_data.shape[-1] == 4
            frame_255 = np.clip(frame_data * 255.0, 0, 255).astype(np.uint8)

            # --- Apply mask overlay if available ---
            if masks_array is not None and idx < len(masks_array):
                mask = masks_array[idx]
                if np.any(mask > 0.5):
                    mask_3d = np.expand_dims(mask, axis=-1)

                    if has_alpha and frame_255.shape[-1] == 4:
                        # For RGBA images, add red tint to show mask while preserving transparency
                        # Use vectorized operations for speed

                        # Get the alpha channel
                        alpha_channel = frame_255[:, :, 3].astype(np.float32)

                        # Convert RGB channels to float for blending
                        frame_float = frame_255[:, :, :3].astype(np.float32)

                        # Blend red tint where mask is active
                        red_mix = 0.5
                        red = np.array([255.0, 0.0, 0.0], dtype=np.float32)

                        # Add red tint to RGB
                        frame_float = np.where(
                            mask_3d > 0.5,
                            np.clip(
                                frame_float + (red - frame_float) * red_mix, 0, 255
                            ),
                            frame_float,
                        )

                        # For transparent areas in the mask, boost alpha to make red visible
                        alpha_channel = np.where(
                            (mask[:, :] > 0.5) & (alpha_channel < 128),
                            np.maximum(alpha_channel, 128),
                            alpha_channel,
                        )

                        # Update the frame
                        frame_255[:, :, :3] = np.clip(frame_float, 0, 255).astype(
                            np.uint8
                        )
                        frame_255[:, :, 3] = np.clip(alpha_channel, 0, 255).astype(
                            np.uint8
                        )
                    else:
                        # For RGB images, use simple addWeighted
                        red_overlay = np.zeros_like(frame_255)
                        red_overlay[:, :, 0] = 255  # Red channel

                        # Blend frame with red overlay using 0.5 alpha (50% transparency)
                        alpha = 0.5
                        frame_255 = np.where(
                            mask_3d > 0.5,
                            cv2.addWeighted(
                                frame_255, 1 - alpha, red_overlay, alpha, 0
                            ),
                            frame_255,
                        ).astype(np.uint8)
            # --- End overlay ---

            if frame_255.ndim == 3 and frame_255.shape[2] == 1:
                frame_255 = frame_255[:, :, 0]

            # Create PIL image with proper mode
            if has_alpha and frame_255.shape[-1] == 4:
                pil_img = Image.fromarray(frame_255, mode="RGBA")
            else:
                pil_img = Image.fromarray(frame_255)

            buffer = BytesIO()
            # Use WebP for better compression and faster transfer
            # Quality 70 with fast method (0) for quick preview generation
            # This significantly reduces file size and encoding time
            pil_img.save(buffer, format="WEBP", quality=70, method=0)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            frames_payload.append(
                {
                    "index": idx,
                    "data": encoded,
                    "width": pil_img.width,
                    "height": pil_img.height,
                }
            )

        response_payload: dict[str, object] = {
            "frames": frames_payload,
            "fps": processing_result["effective_fps"],
            "original_fps": processing_result["original_fps"],
            "selected_frame_indices": processing_result["selected_indices"],
            "frame_count": processing_result["total_frames"],
        }

        if cache_key:
            _maybe_cache_preview(cache_key, response_payload)

        return aiohttp_web.json_response(response_payload)


def _register_mask_route():
    if PromptServer is None or web is None:
        return

    server = PromptServer
    aiohttp_web = web

    @server.instance.routes.post("/videomaskeditor/setmask")
    async def video_mask_editor_setmask(request: _Request):  # pylint: disable=unused-variable
        try:
            data = await request.json()
            node_id = data.get("node_id")
            mask_region_payload = data.get("mask_region")
            region = MaskRegion.from_payload(
                mask_region_payload
                if isinstance(mask_region_payload, Mapping)
                else None
            )
            video = data.get("video")

            if node_id is None:
                return aiohttp_web.json_response(
                    {"error": "Missing node_id"}, status=400
                )

            node_id_str = str(node_id)
            region = region if region else MaskRegion(0, 0, 0, 0)
            _mask_regions[node_id_str] = region
            if video:
                _mask_regions_by_video[str(video)] = region

            _log(f"Mask region set for node {node_id_str}: {region}")
            _log(f"_mask_regions after setting: {_mask_regions}")
            _log(f"_mask_regions dict id: {id(_mask_regions)}")
            _ = _increment_mask_version(node_id_str)

            # Notify the frontend that this node needs to be re-executed
            _ = server.instance.send_sync(
                "videomaskeditor.mask_updated", {"node_id": node_id_str}
            )

            return aiohttp_web.json_response({"success": True})
        except Exception as exc:
            _log(f"Error setting mask: {exc}")
            return aiohttp_web.json_response({"error": str(exc)}, status=500)


def _register_clear_mask_route():
    if PromptServer is None or web is None:
        return

    server = PromptServer
    aiohttp_web = web

    @server.instance.routes.post("/videomaskeditor/clearmask")
    async def video_mask_editor_clearmask(request: _Request):  # pylint: disable=unused-variable
        try:
            data = await request.json()
            node_id = data.get("node_id")

            if node_id is None:
                return aiohttp_web.json_response(
                    {"error": "Missing node_id"}, status=400
                )

            node_id_str = str(node_id)
            if node_id_str in _mask_regions:
                del _mask_regions[node_id_str]
                _log(f"Cleared mask region for node {node_id_str}")
            _ = _increment_mask_version(node_id_str)

            return aiohttp_web.json_response({"success": True})
        except Exception as exc:
            _log(f"Error clearing mask: {exc}")
            return aiohttp_web.json_response({"error": str(exc)}, status=500)


def _register_keyframe_routes():
    """Register routes for keyframe-based masking."""
    if PromptServer is None or web is None:
        return

    server = PromptServer
    aiohttp_web = web

    @server.instance.routes.post("/videomaskeditor/setkeyframe")
    async def video_mask_editor_setkeyframe(request: _Request):
        """Set a mask keyframe for a specific frame."""
        try:
            data = await request.json()
            node_id = data.get("node_id")
            frame_index = data.get("frame_index")
            mask_type = data.get("type", "bbox")  # "bbox", "painted", or "hybrid"
            bbox_data = data.get("bbox")  # bbox dict
            mask_data = data.get("mask_data")  # base64 painted mask

            if node_id is None or frame_index is None:
                return aiohttp_web.json_response(
                    {"error": "Missing node_id or frame_index"}, status=400
                )

            node_id_str = str(node_id)
            if node_id_str not in _mask_keyframes:
                _mask_keyframes[node_id_str] = {}

            bbox_payload: MaskBbox | None = None
            if isinstance(bbox_data, Mapping):
                bbox_payload = {
                    "x": _coerce_int(bbox_data.get("x", 0)),
                    "y": _coerce_int(bbox_data.get("y", 0)),
                    "width": _coerce_int(bbox_data.get("width", 0)),
                    "height": _coerce_int(bbox_data.get("height", 0)),
                }

            mask_entry: MaskKeyframe = {
                "type": str(mask_type),
            }
            if bbox_payload is not None:
                mask_entry["bbox"] = bbox_payload
            if isinstance(mask_data, str):
                mask_entry["mask_data"] = mask_data
            mask_width = data.get("mask_width")
            mask_height = data.get("mask_height")
            if mask_width is not None:
                mask_entry["mask_width"] = _coerce_int(mask_width)
            if mask_height is not None:
                mask_entry["mask_height"] = _coerce_int(mask_height)

            # Support hybrid keyframes with both bbox and painted data
            _mask_keyframes[node_id_str][str(frame_index)] = mask_entry

            _log(
                f"Set keyframe for node {node_id_str}, frame {frame_index}, type {mask_type}"
            )
            _ = _increment_mask_version(node_id_str)
            _save_keyframes_to_disk()

            # Notify frontend
            _ = server.instance.send_sync(
                "videomaskeditor.mask_updated", {"node_id": node_id_str}
            )

            return aiohttp_web.json_response({"success": True})
        except Exception as exc:
            _log(f"Error setting keyframe: {exc}")
            return aiohttp_web.json_response({"error": str(exc)}, status=500)

    @server.instance.routes.post("/videomaskeditor/deletekeyframe")
    async def video_mask_editor_deletekeyframe(request: _Request):
        """Delete a specific keyframe."""
        try:
            data = await request.json()
            node_id = data.get("node_id")
            frame_index = data.get("frame_index")

            if node_id is None or frame_index is None:
                return aiohttp_web.json_response(
                    {"error": "Missing node_id or frame_index"}, status=400
                )

            node_id_str = str(node_id)
            if (
                node_id_str in _mask_keyframes
                and str(frame_index) in _mask_keyframes[node_id_str]
            ):
                del _mask_keyframes[node_id_str][str(frame_index)]
                _log(f"Deleted keyframe for node {node_id_str}, frame {frame_index}")
                _ = _increment_mask_version(node_id_str)
                _save_keyframes_to_disk()

                _ = server.instance.send_sync(
                    "videomaskeditor.mask_updated", {"node_id": node_id_str}
                )

            return aiohttp_web.json_response({"success": True})
        except Exception as exc:
            _log(f"Error deleting keyframe: {exc}")
            return aiohttp_web.json_response({"error": str(exc)}, status=500)

    @server.instance.routes.get("/videomaskeditor/getkeyframes")
    async def video_mask_editor_getkeyframes(request: _Request):
        """Get all keyframes for a node."""
        try:
            params = request.rel_url.query
            node_id = params.get("node_id")

            if not node_id:
                return aiohttp_web.json_response(
                    {"error": "Missing node_id"}, status=400
                )

            node_id_str = str(node_id)
            keyframes = _mask_keyframes.get(node_id_str, {})

            return aiohttp_web.json_response({"keyframes": keyframes})
        except Exception as exc:
            _log(f"Error getting keyframes: {exc}")
            return aiohttp_web.json_response({"error": str(exc)}, status=500)

    @server.instance.routes.get("/videomaskeditor/calculate_wan_cap")
    async def video_mask_editor_calculate_wan_cap(request: _Request):
        """Calculate the appropriate WAN frame cap based on video parameters."""
        try:
            params = request.rel_url.query
            video_name = params.get("video")
            framerate = _coerce_int(params.get("framerate", 0))
            skip_first_frames = _coerce_int(params.get("skip_first_frames", 0))
            select_every_nth = max(1, _coerce_int(params.get("select_every_nth", 1)))

            if not video_name:
                return aiohttp_web.json_response(
                    {"error": "Missing video parameter"}, status=400
                )

            if not folder_paths.exists_annotated_filepath(video_name):
                return aiohttp_web.json_response(
                    {"error": "Video not found"}, status=404
                )

            video_path = folder_paths.get_annotated_filepath(video_name)
            source_total_frames = 0

            # Determine total frames
            if os.path.isdir(video_path):
                image_files = [
                    f
                    for f in os.listdir(video_path)
                    if f.split(".")[-1].lower() in IMAGE_EXTENSIONS
                ]
                source_total_frames = len(image_files)
                # For image folders, assume 18 fps
                assumed_fps = 18.0
                frame_step = (
                    max(1, int(round(assumed_fps / framerate)))
                    if framerate and framerate > 0
                    else 1
                )
            elif os.path.isfile(video_path):
                video_cap = None
                try:
                    video_cap = cv2.VideoCapture(video_path)
                    if not video_cap.isOpened():
                        return aiohttp_web.json_response(
                            {"error": "Could not open video"}, status=500
                        )
                    source_total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    original_fps = video_cap.get(cv2.CAP_PROP_FPS)
                    frame_step = (
                        max(1, int(round(original_fps / framerate)))
                        if framerate and framerate > 0 and original_fps > 0
                        else 1
                    )
                finally:
                    if video_cap is not None:
                        video_cap.release()
            else:
                return aiohttp_web.json_response(
                    {"error": "Invalid video path"}, status=400
                )

            # Calculate available frames
            combined_step = max(1, select_every_nth) * frame_step
            available_frames = max(0, source_total_frames - skip_first_frames)
            available_sampled = (available_frames + combined_step - 1) // combined_step

            # Calculate WAN cap
            if available_sampled < 5:
                wan_cap = -1  # Invalid - not enough frames
            else:
                # WAN formula: 4n+1 where n>=1
                n = (available_sampled - 1) // 4
                if n < 1:
                    wan_cap = -1
                else:
                    wan_cap = 4 * n + 1

            return aiohttp_web.json_response(
                {
                    "wan_cap": wan_cap,
                    "available_sampled": available_sampled,
                    "source_total_frames": source_total_frames,
                    "frame_step": frame_step,
                    "combined_step": combined_step,
                }
            )
        except Exception as exc:
            _log(f"Error calculating WAN cap: {exc}")
            return aiohttp_web.json_response({"error": str(exc)}, status=500)

    @server.instance.routes.post("/videomaskeditor/restorekeyframes")
    async def video_mask_editor_restorekeyframes(request: _Request):
        """Restore keyframes to a previous state (used for cancel functionality)."""
        try:
            data = await request.json()
            node_id = data.get("node_id")
            keyframes_raw = data.get("keyframes", {})

            if node_id is None:
                return aiohttp_web.json_response(
                    {"error": "Missing node_id"}, status=400
                )

            node_id_str = str(node_id)
            keyframes: dict[str, MaskKeyframe] = {}
            if isinstance(keyframes_raw, dict):
                for key, value in keyframes_raw.items():
                    if not isinstance(value, Mapping):
                        continue
                    restored_entry: MaskKeyframe = {}
                    restored_type = value.get("type")
                    if isinstance(restored_type, str):
                        restored_entry["type"] = restored_type
                    restored_bbox = value.get("bbox")
                    if isinstance(restored_bbox, Mapping):
                        restored_entry["bbox"] = {
                            "x": _coerce_int(restored_bbox.get("x", 0)),
                            "y": _coerce_int(restored_bbox.get("y", 0)),
                            "width": _coerce_int(restored_bbox.get("width", 0)),
                            "height": _coerce_int(restored_bbox.get("height", 0)),
                        }
                    restored_mask_data = value.get("mask_data")
                    if isinstance(restored_mask_data, str):
                        restored_entry["mask_data"] = restored_mask_data
                    if "mask_width" in value:
                        restored_entry["mask_width"] = _coerce_int(
                            value.get("mask_width")
                        )
                    if "mask_height" in value:
                        restored_entry["mask_height"] = _coerce_int(
                            value.get("mask_height")
                        )
                    keyframes[str(key)] = restored_entry

            # Restore the keyframes to the provided state
            _mask_keyframes[node_id_str] = keyframes

            _log(f"Restored keyframes for node {node_id_str} (count: {len(keyframes)})")
            _ = _increment_mask_version(node_id_str)
            _save_keyframes_to_disk()

            # Notify frontend
            _ = server.instance.send_sync(
                "videomaskeditor.mask_updated", {"node_id": node_id_str}
            )

            return aiohttp_web.json_response({"success": True})
        except Exception as exc:
            _log(f"Error restoring keyframes: {exc}")
            return aiohttp_web.json_response({"error": str(exc)}, status=500)


_register_preview_route()
_register_mask_route()
_register_clear_mask_route()
_register_keyframe_routes()

# Load keyframes from disk on startup
_load_keyframes_from_disk()


class WANFrameCalculatorNode:
    """Calculate nearest WAN-compatible frame count (1 + 4x)."""

    RETURN_TYPES: tuple[str, ...] = ("INT",)
    RETURN_NAMES: tuple[str, ...] = ("wan_frames",)
    FUNCTION: str = "calculate_wan_frames"
    CATEGORY: str = "animation/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10000,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "rounding_mode": (
                    ["nearest", "max", "min"],
                    {
                        "default": "nearest",
                    },
                ),
            }
        }

    def calculate_wan_frames(self, frame_count: int, rounding_mode: str):
        if frame_count <= 1:
            return (1,)

        if rounding_mode == "max":
            wan_frames = 1 + (int(np.ceil((frame_count - 1) / 4)) * 4)
        elif rounding_mode == "min":
            wan_frames = 1 + (int(np.floor((frame_count - 1) / 4)) * 4)
        else:  # nearest
            wan_frames = 1 + (round((frame_count - 1) / 4) * 4)

        _log(
            f"Input frames: {frame_count} → WAN frames ({rounding_mode}): {wan_frames}"
        )
        return (wan_frames,)


class WANAnimateToVideoPoseStrengthNode:
    """WanAnimateToVideo variant with explicit pose strength scaling."""

    RETURN_TYPES: tuple[str, ...] = (
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES: tuple[str, ...] = (
        "positive",
        "negative",
        "latent",
        "trim_latent",
        "trim_image",
        "video_frame_offset",
    )
    FUNCTION: str = "animate_to_video"
    CATEGORY: str = "conditioning/video_models"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": (
                    "INT",
                    {"default": 832, "min": 16, "max": DIMMAX, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 480, "min": 16, "max": DIMMAX, "step": 16},
                ),
                "length": ("INT", {"default": 77, "min": 1, "max": DIMMAX, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "continue_motion_max_frames": (
                    "INT",
                    {"default": 5, "min": 1, "max": DIMMAX, "step": 4},
                ),
                "video_frame_offset": (
                    "INT",
                    {"default": 0, "min": 0, "max": DIMMAX, "step": 1},
                ),
                "pose_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001},
                ),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "reference_image": ("IMAGE",),
                "face_video": ("IMAGE",),
                "pose_video": ("IMAGE",),
                "background_video": ("IMAGE",),
                "character_mask": ("MASK",),
                "continue_motion": ("IMAGE",),
            },
        }

    def animate_to_video(
        self,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        continue_motion_max_frames,
        video_frame_offset,
        pose_strength,
        clip_vision_output=None,
        reference_image=None,
        face_video=None,
        pose_video=None,
        continue_motion=None,
        background_video=None,
        character_mask=None,
    ):
        trim_to_pose_video = False
        latent_length = ((length - 1) // 4) + 1
        latent_width = width // 8
        latent_height = height // 8
        trim_latent = 0

        if reference_image is None:
            reference_image = torch.zeros((1, height, width, 3))

        image = comfy.utils.common_upscale(
            reference_image[:length].movedim(-1, 1),
            width,
            height,
            "area",
            "center",
        ).movedim(1, -1)
        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = torch.zeros(
            (
                1,
                4,
                concat_latent_image.shape[-3],
                concat_latent_image.shape[-2],
                concat_latent_image.shape[-1],
            ),
            device=concat_latent_image.device,
            dtype=concat_latent_image.dtype,
        )
        trim_latent += concat_latent_image.shape[2]
        ref_motion_latent_length = 0

        if continue_motion is None:
            image = torch.ones((length, height, width, 3)) * 0.5
        else:
            continue_motion = continue_motion[-continue_motion_max_frames:]
            video_frame_offset -= continue_motion.shape[0]
            video_frame_offset = max(0, video_frame_offset)
            continue_motion = comfy.utils.common_upscale(
                continue_motion[-length:].movedim(-1, 1),
                width,
                height,
                "area",
                "center",
            ).movedim(1, -1)
            image = (
                torch.ones(
                    (length, height, width, continue_motion.shape[-1]),
                    device=continue_motion.device,
                    dtype=continue_motion.dtype,
                )
                * 0.5
            )
            image[: continue_motion.shape[0]] = continue_motion
            ref_motion_latent_length += ((continue_motion.shape[0] - 1) // 4) + 1

        if clip_vision_output is not None:
            positive = conditioning_set_values(
                positive, {"clip_vision_output": clip_vision_output}
            )
            negative = conditioning_set_values(
                negative, {"clip_vision_output": clip_vision_output}
            )

        if pose_video is not None:
            if pose_video.shape[0] <= video_frame_offset:
                pose_video = None
            else:
                pose_video = pose_video[video_frame_offset:]

        if pose_video is not None:
            pose_video = comfy.utils.common_upscale(
                pose_video[:length].movedim(-1, 1),
                width,
                height,
                "area",
                "center",
            ).movedim(1, -1)
            if not trim_to_pose_video and pose_video.shape[0] < length:
                pose_video = torch.cat(
                    (pose_video,) + (pose_video[-1:],) * (length - pose_video.shape[0]),
                    dim=0,
                )

            pose_video_latent = vae.encode(pose_video[:, :, :, :3])
            # Scale the latent directly since comfy core model doesn't use pose_strength param
            strength = max(0.0, float(pose_strength))
            if strength != 1.0:
                pose_video_latent = pose_video_latent * strength
            positive = conditioning_set_values(
                positive,
                {"pose_video_latent": pose_video_latent},
            )
            negative = conditioning_set_values(
                negative,
                {"pose_video_latent": pose_video_latent},
            )

            if trim_to_pose_video:
                latent_length = pose_video_latent.shape[2]
                length = latent_length * 4 - 3
                image = image[:length]

        if face_video is not None:
            if face_video.shape[0] <= video_frame_offset:
                face_video = None
            else:
                face_video = face_video[video_frame_offset:]

        if face_video is not None:
            face_video = (
                comfy.utils.common_upscale(
                    face_video[:length].movedim(-1, 1),
                    512,
                    512,
                    "area",
                    "center",
                )
                * 2.0
                - 1.0
            )
            face_video = face_video.movedim(0, 1).unsqueeze(0)
            positive = conditioning_set_values(
                positive, {"face_video_pixels": face_video}
            )
            negative = conditioning_set_values(
                negative, {"face_video_pixels": face_video * 0.0 - 1.0}
            )

        ref_images_num = max(0, ref_motion_latent_length * 4 - 3)
        if background_video is not None:
            if background_video.shape[0] > video_frame_offset:
                background_video = background_video[video_frame_offset:]
                background_video = comfy.utils.common_upscale(
                    background_video[:length].movedim(-1, 1),
                    width,
                    height,
                    "area",
                    "center",
                ).movedim(1, -1)
                if background_video.shape[0] > ref_images_num:
                    image[ref_images_num : background_video.shape[0]] = (
                        background_video[ref_images_num:]
                    )

        mask_refmotion = torch.ones(
            (
                1,
                1,
                latent_length * 4,
                concat_latent_image.shape[-2],
                concat_latent_image.shape[-1],
            ),
            device=mask.device,
            dtype=mask.dtype,
        )
        if continue_motion is not None:
            mask_refmotion[:, :, : ref_motion_latent_length * 4] = 0.0

        if character_mask is not None:
            if (
                character_mask.shape[0] > video_frame_offset
                or character_mask.shape[0] == 1
            ):
                if character_mask.shape[0] == 1:
                    character_mask = character_mask.repeat(
                        (length,) + (1,) * (character_mask.ndim - 1)
                    )
                else:
                    character_mask = character_mask[video_frame_offset:]
                if character_mask.ndim == 3:
                    character_mask = character_mask.unsqueeze(1)
                    character_mask = character_mask.movedim(0, 1)
                if character_mask.ndim == 4:
                    character_mask = character_mask.unsqueeze(1)
                character_mask = comfy.utils.common_upscale(
                    character_mask[:, :, :length],
                    concat_latent_image.shape[-1],
                    concat_latent_image.shape[-2],
                    "nearest-exact",
                    "center",
                )
                if character_mask.shape[2] > ref_images_num:
                    mask_refmotion[:, :, ref_images_num : character_mask.shape[2]] = (
                        character_mask[:, :, ref_images_num:]
                    )

        concat_latent_image = torch.cat(
            (concat_latent_image, vae.encode(image[:, :, :, :3])), dim=2
        )

        mask_refmotion = mask_refmotion.view(
            1,
            mask_refmotion.shape[2] // 4,
            4,
            mask_refmotion.shape[3],
            mask_refmotion.shape[4],
        ).transpose(1, 2)
        mask = torch.cat((mask, mask_refmotion), dim=2)
        positive = conditioning_set_values(
            positive,
            {"concat_latent_image": concat_latent_image, "concat_mask": mask},
        )
        negative = conditioning_set_values(
            negative,
            {"concat_latent_image": concat_latent_image, "concat_mask": mask},
        )

        latent = torch.zeros(
            [batch_size, 16, latent_length + trim_latent, latent_height, latent_width],
            device=comfy.model_management.intermediate_device(),
        )
        out_latent = {"samples": latent}
        trim_image = max(0, ref_motion_latent_length * 4 - 3)
        return (
            positive,
            negative,
            out_latent,
            trim_latent,
            trim_image,
            video_frame_offset + length,
        )


class ReplaceAlpha:
    """Replace alpha channel with a color in masked regions."""

    CATEGORY: str = "Video/Masking"
    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("frames", "alpha")
    FUNCTION: str = "replace_alpha"
    OUTPUT_NODE: bool = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "alpha": ("MASK",),
                "mask": ("MASK",),
                "color": ("STRING", {"default": "#FFFFFF"}),
            }
        }

    def replace_alpha(
        self,
        frames: torch.Tensor,
        alpha: torch.Tensor,
        mask: torch.Tensor,
        color: str,
    ):
        """Replace alpha with color in masked regions.

        Args:
            frames: RGB frames tensor (N, H, W, 3)
            alpha: Alpha channel tensor (N, H, W)
            mask: Mask defining regions to process (N, H, W)
            color: Hex color string (e.g., "#FFFFFF")

        Returns:
            Frames (unchanged) and alpha with color applied in masked regions
        """
        # Normalize mask/alpha shapes to (N, H, W)
        alpha_tensor = alpha
        mask_tensor = mask
        if alpha_tensor.ndim == 4 and alpha_tensor.shape[-1] == 1:
            alpha_tensor = alpha_tensor[..., 0]
        if mask_tensor.ndim == 4 and mask_tensor.shape[-1] == 1:
            mask_tensor = mask_tensor[..., 0]

        # Ensure shapes match
        if (
            frames.shape[0] != alpha_tensor.shape[0]
            or frames.shape[0] != mask_tensor.shape[0]
        ):
            raise ValueError(
                f"Frame count mismatch: frames={frames.shape[0]}, alpha={alpha_tensor.shape[0]}, mask={mask_tensor.shape[0]}"
            )

        if (
            frames.shape[1:3] != alpha_tensor.shape[1:3]
            or frames.shape[1:3] != mask_tensor.shape[1:3]
        ):
            raise ValueError(
                f"Frame size mismatch: frames={frames.shape[1:3]}, alpha={alpha_tensor.shape[1:3]}, mask={mask_tensor.shape[1:3]}"
            )

        # Parse hex color to RGB (0-1 range) for compositing.
        r, g, b = parse_hex_color(color, fallback=(255, 255, 255))
        color_rgb = torch.tensor(
            [r / 255.0, g / 255.0, b / 255.0],
            device=frames.device,
            dtype=frames.dtype,
        )

        result_alpha = alpha_tensor.clone()
        result_frames = frames.clone()
        replace_regions = mask_tensor > 0.5
        if replace_regions.any():
            alpha_3d = alpha_tensor.unsqueeze(-1)
            blended = result_frames * alpha_3d + color_rgb * (1.0 - alpha_3d)
            replace_regions_3d = replace_regions.unsqueeze(-1)
            result_frames = torch.where(replace_regions_3d, blended, result_frames)
            result_alpha = torch.where(
                replace_regions, torch.ones_like(result_alpha), result_alpha
            )

        return (result_frames, result_alpha)


class PreviewImageAlpha:
    """Preview images with alpha transparency (no background replacement)."""

    CATEGORY: str = "Video/Masking"
    RETURN_TYPES: tuple[str, ...] = ()
    FUNCTION: str = "preview_alpha"
    OUTPUT_NODE: bool = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "alpha": ("MASK",),
            }
        }

    def preview_alpha(self, frames: torch.Tensor, alpha: torch.Tensor):
        """Preview frames with alpha channel, showing transparency.

        Args:
            frames: RGB frames tensor (N, H, W, 3)
            alpha: Alpha channel tensor (N, H, W)

        Returns:
            Preview data for ComfyUI UI
        """
        # Convert to numpy
        frames_np = frames.cpu().numpy()  # Shape: (N, H, W, 3)
        alpha_np = alpha.cpu().numpy()  # Shape: (N, H, W)

        # Ensure shapes match
        if frames_np.shape[0] != alpha_np.shape[0]:
            raise ValueError(
                f"Frame count mismatch: frames={frames_np.shape[0]}, alpha={alpha_np.shape[0]}"
            )

        if frames_np.shape[1:3] != alpha_np.shape[1:3]:
            raise ValueError(
                f"Frame size mismatch: frames={frames_np.shape[1:3]}, alpha={alpha_np.shape[1:3]}"
            )

        # Prepare output for ComfyUI preview
        results = []

        for i in range(frames_np.shape[0]):
            # Combine RGB and alpha into RGBA
            rgba = np.concatenate([frames_np[i], alpha_np[i][:, :, np.newaxis]], axis=2)

            # Convert to uint8
            rgba_255 = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)

            # Convert to PIL Image with alpha
            pil_img = Image.fromarray(rgba_255, mode="RGBA")

            # Save to output directory
            output_dir = folder_paths.get_output_directory()
            preview_name = f"preview_alpha_{i:04d}.png"
            preview_path = os.path.join(output_dir, preview_name)
            pil_img.save(preview_path)

            results.append(
                {"filename": preview_name, "subfolder": "", "type": "output"}
            )

        return {"ui": {"images": results}}


class BatchImageSave:
    """Save a batch of images with sequential naming."""

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
        """Save batch of images with sequential naming.

        Args:
            images: Image tensor (N, H, W, C)
            path: Subfolder path (relative to output directory)
            filename_prefix: Prefix for filenames
            delimiter: Character(s) between prefix and number
            extension: File extension (png, jpg, jpeg, webp)
        Returns:
            UI response with saved file information and outputs
        """
        # Convert to numpy
        images_np = images.cpu().numpy()

        if images_np.ndim != 4:
            raise ValueError(
                f"Expected images with shape (N, H, W, C), got shape {images_np.shape}"
            )

        num_images = images_np.shape[0]
        saved_files = []

        # Get ComfyUI output directory as base
        output_dir = folder_paths.get_output_directory()

        # Normalize extension
        ext = extension.lower()
        if ext == "jpg":
            ext = "jpeg"

        # Determine PIL format
        format_map = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
        pil_format = format_map.get(ext, "PNG")

        # Create full directory path, reusing the same resolved folder across list-index calls.
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

        # Find the next available index to avoid overwriting existing files
        existing_max = 0
        pattern = re.compile(
            rf"^{re.escape(filename_prefix)}{re.escape(delimiter)}(\d+)\.{re.escape(ext)}$"
        )
        for name in os.listdir(full_dir):
            match = pattern.match(name)
            if match:
                existing_max = max(existing_max, int(match.group(1)))

        start_index = existing_max + 1

        # Determine number of digits for padding based on final count (min 2)
        digits = max(2, len(str(start_index + num_images - 1)))

        for i in range(num_images):
            # Create sequential filename: prefix + delimiter + number + extension
            index = start_index + i
            filename = f"{filename_prefix}{delimiter}{index:0{digits}d}.{ext}"

            # Relative path for tracking
            relative_path = os.path.join(path, filename) if path else filename

            # Full path for saving
            full_path = os.path.join(full_dir, filename)

            # Convert frame to PIL Image
            frame = images_np[i]
            frame = np.clip(frame, 0.0, 1.0)

            # Handle different channel counts
            if frame.shape[-1] == 1:
                # Grayscale
                frame_255 = (frame[:, :, 0] * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_255, mode="L")
            elif frame.shape[-1] == 3:
                # RGB
                frame_255 = (frame * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_255, mode="RGB")
            elif frame.shape[-1] == 4:
                # RGBA
                frame_255 = (frame * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_255, mode="RGBA")
                # Convert RGBA to RGB for JPEG (no alpha support)
                if pil_format == "JPEG":
                    pil_img = pil_img.convert("RGB")
            else:
                raise ValueError(f"Unsupported number of channels: {frame.shape[-1]}")

            # Save the image
            pil_img.save(full_path, format=pil_format)
            saved_files.append(relative_path)
            _log(f"Saved image {index}/{num_images} to {full_path}")

        # Determine the folder path (relative to output folder)
        folder_path = resolved_path if resolved_path else ""

        # Join file names with newlines
        file_names = "\n".join([os.path.basename(f) for f in saved_files])

        return {
            "ui": {
                "text": [f"Saved {num_images} images to {resolved_path or 'output'}"]
            },
            "result": (folder_path, file_names),
        }


class SaveImageSequenceZip:
    """Save connected image/mask sequences as files in a ZIP archive."""

    CATEGORY: str = "Video/Masking"
    RETURN_TYPES: tuple[str, ...] = ()
    FUNCTION: str = "save_sequence"
    OUTPUT_NODE: bool = True
    INPUT_IS_LIST: bool = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zip_path": (
                    "STRING",
                    {"default": "output/archive", "multiline": False},
                ),
            },
            "optional": {
                "input1_prefix": ("STRING", {"default": ""}),
                "input2_prefix": ("STRING", {"default": ""}),
                "input3_prefix": ("STRING", {"default": ""}),
                "input4_prefix": ("STRING", {"default": ""}),
                "input5_prefix": ("STRING", {"default": ""}),
                "input6_prefix": ("STRING", {"default": ""}),
                "input7_prefix": ("STRING", {"default": ""}),
                "input8_prefix": ("STRING", {"default": ""}),
            },
        }

    def save_sequence(self, zip_path: str, **kwargs: object):
        """Save connected image/mask sequences in a ZIP file."""
        if isinstance(zip_path, (list, tuple)):
            zip_path = zip_path[0] if zip_path else ""

        inputs = []
        for index in range(1, 9):
            data = kwargs.get(f"input{index}")
            if data is None:
                continue
            prefix_value = kwargs.get(f"input{index}_prefix", f"input{index}")
            if isinstance(prefix_value, (list, tuple)):
                prefix_value = prefix_value[0] if prefix_value else ""
            prefix_text = prefix_value if isinstance(prefix_value, str) else ""
            prefix_text = prefix_text.strip() or f"input{index}"
            inputs.append((f"input{index}", data, prefix_text))

        # Process zip_path: add .zip extension if not present
        zip_path = zip_path.strip()

        def format_path(template: str, idx: int) -> str:
            if "{index" in template:
                return template.format_map({"index": idx})
            return template.format(idx)

        # Get output directory and construct full path
        output_dir = folder_paths.get_output_directory()

        uses_template = False
        if "{" in zip_path and "}" in zip_path:
            try:
                test_candidate = format_path(zip_path, 1)
                uses_template = test_candidate != zip_path
            except (ValueError, KeyError, IndexError):
                uses_template = False

        if uses_template:
            index = 1
            while True:
                try:
                    candidate = format_path(zip_path, index)
                except (ValueError, KeyError, IndexError):
                    uses_template = False
                    break
                if not candidate.endswith(".zip"):
                    candidate = f"{candidate}.zip"
                full_candidate = (
                    candidate
                    if os.path.isabs(candidate)
                    else os.path.join(output_dir, candidate)
                )
                if not os.path.exists(full_candidate):
                    zip_path = candidate
                    full_zip_path = full_candidate
                    break
                index += 1

        if not uses_template:
            if not zip_path.endswith(".zip"):
                zip_path = f"{zip_path}.zip"
            if os.path.isabs(zip_path):
                full_zip_path = zip_path
            else:
                full_zip_path = os.path.join(output_dir, zip_path)

        # Create parent directory if it doesn't exist
        zip_dir = os.path.dirname(full_zip_path)
        if zip_dir:
            os.makedirs(zip_dir, exist_ok=True)

        _log(f"Saving ZIP to: {full_zip_path}")

        # Create ZIP file
        used_names = set()

        def reserve_name(file_name: str) -> str:
            if file_name not in used_names:
                used_names.add(file_name)
                return file_name
            base, ext = os.path.splitext(file_name)
            counter = 1
            while True:
                candidate = f"{base}_{counter:02d}{ext}"
                if candidate not in used_names:
                    used_names.add(candidate)
                    return candidate
                counter += 1

        with zipfile.ZipFile(full_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for _input_name, data, prefix in inputs:
                if data is None:
                    continue
                if isinstance(data, str):
                    payload = data.encode("utf-8")
                    ext = self._extension_for_text(data)
                    file_name = reserve_name(f"{prefix}.{ext}")
                    _ = zipf.writestr(file_name, payload)
                    continue
                if isinstance(data, dict):
                    payload = json.dumps(data, indent=2).encode("utf-8")
                    file_name = reserve_name(f"{prefix}.json")
                    _ = zipf.writestr(file_name, payload)
                    continue
                if isinstance(data, (list, tuple)) and data:
                    if all(isinstance(item, str) for item in data):
                        for i, item in enumerate(data, start=1):
                            payload = item.encode("utf-8")
                            ext = self._extension_for_text(item)
                            file_name = reserve_name(f"{prefix}.{ext}")
                            _ = zipf.writestr(file_name, payload)
                        continue
                    if all(isinstance(item, dict) for item in data):
                        for i, item in enumerate(data, start=1):
                            payload = json.dumps(item, indent=2).encode("utf-8")
                            file_name = reserve_name(f"{prefix}.json")
                            _ = zipf.writestr(file_name, payload)
                        continue
                frames = self._normalize_frames(data)

                # Check if prefix has an extension
                prefix_base, prefix_ext = os.path.splitext(prefix)
                has_extension = bool(prefix_ext)

                # Determine format and extension
                if has_extension:
                    # Use extension from prefix
                    ext = prefix_ext.lstrip(".")
                    format_name = (
                        ext.upper()
                        if ext.upper() in {"PNG", "JPEG", "JPG", "WEBP"}
                        else "PNG"
                    )
                    if format_name == "JPG":
                        format_name = "JPEG"
                else:
                    # Default to PNG
                    ext = "png"
                    format_name = "PNG"

                if frames.shape[0] == 1:
                    pil_img = self._to_pil(frames[0], ext)
                    img_buffer = BytesIO()
                    pil_img.save(img_buffer, format=format_name)
                    _ = img_buffer.seek(0)

                    image_filename = reserve_name(
                        f"{prefix_base}{prefix_ext}"
                        if has_extension
                        else f"{prefix_base}.{ext}"
                    )

                    _ = zipf.writestr(image_filename, img_buffer.getvalue())
                else:
                    # For batches, use format string if present, otherwise use global index
                    for i in range(frames.shape[0]):
                        frame_index = i + 1
                        pil_img = self._to_pil(frames[i], ext)
                        img_buffer = BytesIO()
                        pil_img.save(img_buffer, format=format_name)
                        _ = img_buffer.seek(0)

                        # Check if prefix contains format string (e.g., {:02d})
                        if "{" in prefix_base:
                            try:
                                image_filename = prefix_base.format(frame_index) + (
                                    prefix_ext if has_extension else f".{ext}"
                                )
                            except (KeyError, IndexError):
                                # If format string fails, fall back to default naming
                                digits = max(2, len(str(frames.shape[0])))
                                image_filename = (
                                    f"{prefix}_{frame_index:0{digits}d}.{ext}"
                                    if not has_extension
                                    else f"{prefix_base}_{frame_index:0{digits}d}{prefix_ext}"
                                )
                        else:
                            digits = max(2, len(str(frames.shape[0])))
                            image_filename = (
                                f"{prefix}_{frame_index:0{digits}d}.{ext}"
                                if not has_extension
                                else f"{prefix_base}_{frame_index:0{digits}d}{prefix_ext}"
                            )

                        image_filename = reserve_name(image_filename)
                        _ = zipf.writestr(image_filename, img_buffer.getvalue())

        _log(f"Saved ZIP to {full_zip_path}")

        # Extract just the filename for the download URL
        zip_filename = os.path.basename(zip_path)

        # Remove "output/" prefix from zip_path for the download URL (with optional leading/trailing slash)
        zip_path_for_url = zip_path.strip("/")
        if zip_path_for_url.startswith("output/"):
            zip_path_for_url = zip_path_for_url[len("output/") :]
        elif zip_path_for_url.startswith("output"):
            zip_path_for_url = zip_path_for_url[len("output") :].lstrip("/")

        # Create download URL (ComfyUI's /view endpoint)
        download_url = f"/view?filename={zip_path_for_url}&type=output"

        # Return UI with download link as HTML
        return {
            "ui": {
                "text": [
                    f'<a href="{download_url}" target="_blank" style="color: #4a9eff; text-decoration: underline;">Download: {zip_filename}</a>'
                ],
            }
        }

    @staticmethod
    def _extension_for_text(text: str) -> str:
        trimmed = text.lstrip()
        if trimmed.startswith("{") or trimmed.startswith("["):
            return "json"
        return "txt"

    @staticmethod
    def _normalize_frames(data: torch.Tensor | np.ndarray | list | tuple) -> np.ndarray:
        if isinstance(data, (list, tuple)):
            frames_list = [
                SaveImageSequenceZip._normalize_frames(item) for item in data
            ]
            if not frames_list:
                raise ValueError("Expected non-empty list/tuple for frames")
            return np.concatenate(frames_list, axis=0)
        if isinstance(data, np.ndarray):
            frames = torch.from_numpy(data).float()
        else:
            frames = data.detach().cpu().float()
        if frames.ndim == 2:
            frames = frames.unsqueeze(0).unsqueeze(-1)
        elif frames.ndim == 3:
            if frames.shape[-1] in {1, 3, 4}:
                frames = frames.unsqueeze(0)
            else:
                frames = frames.unsqueeze(-1)
        elif frames.ndim == 4 and frames.shape[-1] not in {1, 3, 4}:
            frames = frames.unsqueeze(-1)
        if frames.ndim != 4:
            raise ValueError("Expected data with shape (N, H, W, C)")
        return frames.numpy()

    @staticmethod
    def _to_pil(frame: np.ndarray, ext: str) -> Image.Image:
        frame = np.clip(frame, 0.0, 1.0)
        if frame.shape[-1] == 1:
            frame_255 = (frame[..., 0] * 255).astype(np.uint8)
            return Image.fromarray(frame_255, mode="L")
        if frame.shape[-1] == 4 and ext in {"jpg", "jpeg"}:
            frame = frame[..., :3]
        frame_255 = (frame * 255).astype(np.uint8)
        mode = "RGBA" if frame_255.shape[-1] == 4 else "RGB"
        return Image.fromarray(frame_255, mode=mode)
