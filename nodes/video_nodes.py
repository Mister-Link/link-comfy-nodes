from __future__ import annotations

import base64
import json
import os
import random
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

import folder_paths

from ..utils import parse_hex_color

try:
    from aiohttp import web

    from server import PromptServer
except Exception:  # pragma: no cover - ComfyUI runtime handles availability
    PromptServer = None
    web = None

VIDEO_EXTENSIONS = ["webm", "mp4", "mkv", "gif", "mov"]
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"]
BIGMAX = 2**53 - 1
DIMMAX = 8192
DEFAULT_PREVIEW_FRAME_LIMIT = 120
PREVIEW_CACHE_MAX_ITEMS = 5  # Increased since WebP is smaller
PREVIEW_CACHE_MAX_BYTES = 20_000_000  # Reduced since WebP is more efficient


def _log(message: str):
    print(f"[VideoMaskEditor] {message}")


@dataclass
class MaskRegion:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_payload(cls, payload: Optional[Dict]) -> Optional["MaskRegion"]:
        if not payload:
            return None
        try:
            return cls(
                x=int(payload.get("x", 0)),
                y=int(payload.get("y", 0)),
                width=int(payload.get("width", 0)),
                height=int(payload.get("height", 0)),
            )
        except Exception:
            return None

    def clamp(self, max_width: int, max_height: int) -> "MaskRegion":
        x = max(0, min(self.x, max_width - 1))
        y = max(0, min(self.y, max_height - 1))
        width = max(0, min(self.width, max_width - x))
        height = max(0, min(self.height, max_height - y))
        return MaskRegion(x=x, y=y, width=width, height=height)


_mask_regions: Dict[str, MaskRegion] = {}
_mask_regions_by_video: Dict[str, MaskRegion] = {}
_mask_versions: Dict[str, int] = {}
_mask_keyframes: Dict[str, Dict] = {}  # node_id -> {frame_index: mask_data}
_preview_cache: "OrderedDict[str, Dict]" = OrderedDict()

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
                _mask_keyframes = json.load(f)
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


def _get_mask_version(node_id: Optional[str]) -> int:
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


def _maybe_cache_preview(key: str, payload: Dict):
    """Cache preview payload when it is reasonably small."""
    try:
        total_bytes = sum(
            len(frame.get("data", "")) for frame in payload.get("frames", [])
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
        _preview_cache.popitem(last=False)


def _generate_masks_from_keyframes(
    node_id: str, frame_count: int, width: int, height: int
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

        for i, kf_idx in enumerate(keyframe_indices):
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

            if next_kf_data and next_kf_data.get("bbox") and next_kf_idx is not None:
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
) -> Tuple[int, int]:
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
    preview_max_frames: Optional[int] = None,
):
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

    frames_list: List[np.ndarray] = []
    selected_frame_indices: List[int] = []

    max_frames = frame_load_cap if frame_load_cap > 0 else None
    if preview_max_frames is not None and preview_max_frames > 0:
        max_frames = (
            min(max_frames, preview_max_frames)
            if max_frames is not None
            else preview_max_frames
        )

    # Process images
    for frame_index, img_path in enumerate(image_files):
        if frame_index < skip_first_frames:
            continue

        relative_index = frame_index - skip_first_frames
        if relative_index % select_every_nth != 0:
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
                img = img.resize((target_width, target_height), Image.LANCZOS)

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

    # For image sequences, we don't have a native FPS, so use a sensible default
    # If framerate is 0 or not specified, default to 18 fps for image folders
    base_fps = framerate if framerate > 0 else 18
    effective_fps = base_fps / max(1, select_every_nth)

    return {
        "frames": frames_list,
        "selected_indices": selected_frame_indices,
        "target_width": target_width,
        "target_height": target_height,
        "original_fps": base_fps,  # Use specified or default FPS
        "effective_fps": effective_fps,
        "total_frames": len(image_files),
        "frame_step": 1,
        "combined_step": select_every_nth,
    }


def _load_video_frames(
    video_path: str,
    framerate: int,
    custom_width: int,
    custom_height: int,
    frame_load_cap: int,
    skip_first_frames: int,
    select_every_nth: int,
    preview_max_frames: Optional[int] = None,
):
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

    frame_step = (
        max(1, int(round(original_fps / framerate)))
        if framerate and original_fps > 0
        else 1
    )
    combined_step = max(1, select_every_nth) * frame_step
    offset_adjustment = 1 if frame_step > 1 else 0

    frames_list: List[np.ndarray] = []
    selected_frame_indices: List[int] = []
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
            if offset_adjustment and relative_index == 0:
                frame_index += 1
                continue

            adjusted_index = relative_index - offset_adjustment
            if adjusted_index < 0 or adjusted_index % combined_step != 0:
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

    base_fps = framerate if framerate != 0 else int(original_fps)
    effective_fps = base_fps / max(1, select_every_nth)

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

    CATEGORY = "Video/Masking"
    RETURN_TYPES = ("IMAGE", "INT", "MASK", "MASK")
    RETURN_NAMES = ("frames", "frame_count", "masks", "alpha_channel")
    FUNCTION = "load_video"
    OUTPUT_NODE = False

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
                "custom_width": (
                    "INT",
                    {"default": 0, "min": 0, "max": DIMMAX, "disable": 0},
                ),
                "custom_height": (
                    "INT",
                    {"default": 0, "min": 0, "max": DIMMAX, "disable": 0},
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
        custom_width: int,
        custom_height: int,
        frame_load_cap: int,
        skip_first_frames: int,
        select_every_nth: int,
        is_wan: bool,
        bg_color: str,
        force_size: str = "",
        unique_id: Optional[str] = None,
    ):
        video_path = folder_paths.get_annotated_filepath(source)
        source_total_frames = 0

        # Determine total frames for WAN snapping logic
        if os.path.isdir(video_path):
            image_files = [
                f
                for f in os.listdir(video_path)
                if f.split(".")[-1].lower() in IMAGE_EXTENSIONS
            ]
            source_total_frames = len(image_files)
        elif os.path.isfile(video_path):
            try:
                video_cap = cv2.VideoCapture(video_path)
                if not video_cap.isOpened():
                    raise ValueError(
                        f"Could not open video for frame count: {video_path}"
                    )
                source_total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                if video_cap:
                    video_cap.release()

        if is_wan and frame_load_cap > 0 and source_total_frames > 0:
            original_cap = frame_load_cap
            # Snap to nearest WAN count
            snapped_cap = 1 + (round((frame_load_cap - 1) / 4) * 4)
            if snapped_cap <= 0:
                snapped_cap = 1

            # If snapped cap is more than we have, snap down to max possible WAN count
            if snapped_cap > source_total_frames:
                snapped_cap = 1 + (int(np.floor((source_total_frames - 1) / 4)) * 4)
                if snapped_cap <= 0:
                    snapped_cap = 1

            if frame_load_cap != snapped_cap:
                _log(
                    f"WAN mode enabled. Snapped frame_load_cap from {original_cap} to {snapped_cap} (total frames: {source_total_frames})"
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
                custom_width,
                custom_height,
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
                custom_width,
                custom_height,
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
            pil_img.save(os.path.join(output_dir, preview_name))

        _log(
            f"Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}"
        )
        json.dumps(selected_frame_indices)  # backward compatibility noop

        # Generate masks from keyframes
        masks_array = _generate_masks_from_keyframes(
            unique_id, frames_tensor.shape[0], target_width, target_height
        )

        masks_tensor = torch.from_numpy(masks_array)
        _log(f"Masks tensor shape: {masks_tensor.shape}, dtype: {masks_tensor.dtype}")
        _log(f"Alpha tensor shape: {alpha_tensor.shape}, dtype: {alpha_tensor.dtype}")

        return frames_tensor, frames_tensor.shape[0], masks_tensor, alpha_tensor

    @classmethod
    def IS_CHANGED(cls, source, **kwargs):
        video_path = folder_paths.get_annotated_filepath(source)
        unique_id = kwargs.get("unique_id")

        # Increment version when video file changes to trigger re-execution
        if unique_id:
            _increment_mask_version(unique_id)

        mask_version = _get_mask_version(unique_id)
        if os.path.exists(video_path):
            return f"{os.path.getmtime(video_path)}_{mask_version}"
        return f"missing_{mask_version}"

    @classmethod
    def VALIDATE_INPUTS(cls, source, **kwargs):
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

    @PromptServer.instance.routes.get("/videomaskeditor/preview")
    async def video_mask_editor_preview(request):  # pylint: disable=unused-variable
        params = request.rel_url.query
        video_name = params.get("video")
        node_id = params.get("node_id")  # Added to get keyframes for this node

        if not video_name:
            return web.json_response({"error": "Missing video parameter"}, status=400)

        if not folder_paths.exists_annotated_filepath(video_name):
            return web.json_response({"error": "Video not found"}, status=404)

        def _int_param(key: str, default: int, minimum: Optional[int] = None) -> int:
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

        # Performance optimization: aggressive downscaling for dialog previews
        preview_downscale = 2 if skip_mask else 1
        if preview_downscale > 1:
            if custom_width > 0:
                custom_width = max(64, custom_width // preview_downscale)
            if custom_height > 0:
                custom_height = max(64, custom_height // preview_downscale)

        mask_version = _get_mask_version(node_id)
        cache_key = None

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
                _log(
                    f"Preview cache HIT for {video_name} ({len(cached_payload.get('frames', []))} frames)"
                )
                return web.json_response(cached_payload)

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
            return web.json_response({"error": str(exc)}, status=400)

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

        frames_payload = []
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
            # Quality 80 provides good balance between size and quality
            pil_img.save(buffer, format="WEBP", quality=80, method=4)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            frames_payload.append(
                {
                    "index": idx,
                    "data": encoded,
                    "width": pil_img.width,
                    "height": pil_img.height,
                }
            )

        response_payload = {
            "frames": frames_payload,
            "fps": processing_result["effective_fps"],
            "original_fps": processing_result["original_fps"],
            "selected_frame_indices": processing_result["selected_indices"],
            "frame_count": processing_result["total_frames"],
        }

        if cache_key:
            _maybe_cache_preview(cache_key, response_payload)

        return web.json_response(response_payload)


def _register_mask_route():
    if PromptServer is None or web is None:
        return

    @PromptServer.instance.routes.post("/videomaskeditor/setmask")
    async def video_mask_editor_setmask(request):  # pylint: disable=unused-variable
        try:
            data = await request.json()
            node_id = data.get("node_id")
            region = MaskRegion.from_payload(data.get("mask_region"))
            video = data.get("video")

            if node_id is None:
                return web.json_response({"error": "Missing node_id"}, status=400)

            region = region if region else MaskRegion(0, 0, 0, 0)
            _mask_regions[str(node_id)] = region
            if video:
                _mask_regions_by_video[str(video)] = region

            _log(f"Mask region set for node {node_id}: {region}")
            _log(f"_mask_regions after setting: {_mask_regions}")
            _log(f"_mask_regions dict id: {id(_mask_regions)}")
            _increment_mask_version(node_id)

            # Notify the frontend that this node needs to be re-executed
            PromptServer.instance.send_sync(
                "videomaskeditor.mask_updated", {"node_id": node_id}
            )

            return web.json_response({"success": True})
        except Exception as exc:
            _log(f"Error setting mask: {exc}")
            return web.json_response({"error": str(exc)}, status=500)


def _register_clear_mask_route():
    if PromptServer is None or web is None:
        return

    @PromptServer.instance.routes.post("/videomaskeditor/clearmask")
    async def video_mask_editor_clearmask(request):  # pylint: disable=unused-variable
        try:
            data = await request.json()
            node_id = data.get("node_id")

            if node_id is None:
                return web.json_response({"error": "Missing node_id"}, status=400)

            if str(node_id) in _mask_regions:
                del _mask_regions[str(node_id)]
                _log(f"Cleared mask region for node {node_id}")
            _increment_mask_version(node_id)

            return web.json_response({"success": True})
        except Exception as exc:
            _log(f"Error clearing mask: {exc}")
            return web.json_response({"error": str(exc)}, status=500)


def _register_keyframe_routes():
    """Register routes for keyframe-based masking."""
    if PromptServer is None or web is None:
        return

    @PromptServer.instance.routes.post("/videomaskeditor/setkeyframe")
    async def video_mask_editor_setkeyframe(request):
        """Set a mask keyframe for a specific frame."""
        try:
            data = await request.json()
            node_id = data.get("node_id")
            frame_index = data.get("frame_index")
            mask_type = data.get("type", "bbox")  # "bbox", "painted", or "hybrid"
            bbox_data = data.get("bbox")  # bbox dict
            mask_data = data.get("mask_data")  # base64 painted mask

            if node_id is None or frame_index is None:
                return web.json_response(
                    {"error": "Missing node_id or frame_index"}, status=400
                )

            node_id = str(node_id)
            if node_id not in _mask_keyframes:
                _mask_keyframes[node_id] = {}

            # Support hybrid keyframes with both bbox and painted data
            _mask_keyframes[node_id][str(frame_index)] = {
                "type": mask_type,
                "bbox": bbox_data,
                "mask_data": mask_data,
                "mask_width": data.get("mask_width"),
                "mask_height": data.get("mask_height"),
            }

            _log(
                f"Set keyframe for node {node_id}, frame {frame_index}, type {mask_type}"
            )
            _increment_mask_version(node_id)
            _save_keyframes_to_disk()

            # Notify frontend
            PromptServer.instance.send_sync(
                "videomaskeditor.mask_updated", {"node_id": node_id}
            )

            return web.json_response({"success": True})
        except Exception as exc:
            _log(f"Error setting keyframe: {exc}")
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.post("/videomaskeditor/deletekeyframe")
    async def video_mask_editor_deletekeyframe(request):
        """Delete a specific keyframe."""
        try:
            data = await request.json()
            node_id = data.get("node_id")
            frame_index = data.get("frame_index")

            if node_id is None or frame_index is None:
                return web.json_response(
                    {"error": "Missing node_id or frame_index"}, status=400
                )

            node_id = str(node_id)
            if (
                node_id in _mask_keyframes
                and str(frame_index) in _mask_keyframes[node_id]
            ):
                del _mask_keyframes[node_id][str(frame_index)]
                _log(f"Deleted keyframe for node {node_id}, frame {frame_index}")
                _increment_mask_version(node_id)
                _save_keyframes_to_disk()

                PromptServer.instance.send_sync(
                    "videomaskeditor.mask_updated", {"node_id": node_id}
                )

            return web.json_response({"success": True})
        except Exception as exc:
            _log(f"Error deleting keyframe: {exc}")
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/videomaskeditor/getkeyframes")
    async def video_mask_editor_getkeyframes(request):
        """Get all keyframes for a node."""
        try:
            params = request.rel_url.query
            node_id = params.get("node_id")

            if not node_id:
                return web.json_response({"error": "Missing node_id"}, status=400)

            node_id = str(node_id)
            keyframes = _mask_keyframes.get(node_id, {})

            return web.json_response({"keyframes": keyframes})
        except Exception as exc:
            _log(f"Error getting keyframes: {exc}")
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.post("/videomaskeditor/restorekeyframes")
    async def video_mask_editor_restorekeyframes(request):
        """Restore keyframes to a previous state (used for cancel functionality)."""
        try:
            data = await request.json()
            node_id = data.get("node_id")
            keyframes = data.get("keyframes", {})

            if node_id is None:
                return web.json_response({"error": "Missing node_id"}, status=400)

            node_id = str(node_id)

            # Restore the keyframes to the provided state
            _mask_keyframes[node_id] = keyframes

            _log(f"Restored keyframes for node {node_id} (count: {len(keyframes)})")
            _increment_mask_version(node_id)
            _save_keyframes_to_disk()

            # Notify frontend
            PromptServer.instance.send_sync(
                "videomaskeditor.mask_updated", {"node_id": node_id}
            )

            return web.json_response({"success": True})
        except Exception as exc:
            _log(f"Error restoring keyframes: {exc}")
            return web.json_response({"error": str(exc)}, status=500)


_register_preview_route()
_register_mask_route()
_register_clear_mask_route()
_register_keyframe_routes()

# Load keyframes from disk on startup
_load_keyframes_from_disk()


class WANFrameCalculatorNode:
    """Calculate nearest WAN-compatible frame count (1 + 4x)."""

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("wan_frames",)
    FUNCTION = "calculate_wan_frames"
    CATEGORY = "animation/utils"

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


class ReplaceAlpha:
    """Replace alpha channel with a color in masked regions."""

    CATEGORY = "Video/Masking"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("frames", "alpha")
    FUNCTION = "replace_alpha"
    OUTPUT_NODE = False

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
            RGB frames with alpha replaced by color in masked regions
        """
        # Parse hex color to RGB (0-1 range)
        color = color.strip()
        if color.startswith("#"):
            color = color[1:]

        try:
            r = int(color[0:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:6], 16) / 255.0
        except (ValueError, IndexError):
            _log(f"Invalid color format: {color}, using white")
            r, g, b = 1.0, 1.0, 1.0

        # Convert to numpy for processing
        frames_np = frames.cpu().numpy()  # Shape: (N, H, W, 3)
        alpha_np = alpha.cpu().numpy()  # Shape: (N, H, W)
        mask_np = mask.cpu().numpy()  # Shape: (N, H, W)

        # Ensure shapes match
        if (
            frames_np.shape[0] != alpha_np.shape[0]
            or frames_np.shape[0] != mask_np.shape[0]
        ):
            raise ValueError(
                f"Frame count mismatch: frames={frames_np.shape[0]}, alpha={alpha_np.shape[0]}, mask={mask_np.shape[0]}"
            )

        if (
            frames_np.shape[1:3] != alpha_np.shape[1:3]
            or frames_np.shape[1:3] != mask_np.shape[1:3]
        ):
            raise ValueError(
                f"Frame size mismatch: frames={frames_np.shape[1:3]}, alpha={alpha_np.shape[1:3]}, mask={mask_np.shape[1:3]}"
            )

        result = frames_np.copy()
        result_alpha = alpha_np.copy()

        # Process each frame
        for i in range(frames_np.shape[0]):
            frame_rgb = frames_np[i]  # (H, W, 3)
            frame_alpha = alpha_np[i]  # (H, W)
            frame_mask = mask_np[i]  # (H, W)

            # Find masked regions (where mask > 0.5)
            masked_regions = frame_mask > 0.5

            # In masked regions, blend RGB with color based on alpha
            # Formula: result = rgb * alpha + color * (1 - alpha)
            if np.any(masked_regions):
                alpha_3d = frame_alpha[:, :, np.newaxis]  # (H, W, 1)
                color_rgb = np.array([r, g, b], dtype=np.float32)  # (3,)

                # Apply blending only where masked
                result[i][masked_regions] = frame_rgb[masked_regions] * alpha_3d[
                    masked_regions
                ] + color_rgb * (1.0 - alpha_3d[masked_regions])

                # Set alpha to 1.0 (fully opaque) in masked regions where we replaced transparency
                result_alpha[i][masked_regions] = 1.0

        # Clip and convert back to tensor
        result = np.clip(result, 0.0, 1.0)
        result_tensor = torch.from_numpy(result).to(frames.device)
        result_alpha_tensor = torch.from_numpy(result_alpha).to(alpha.device)

        return (result_tensor, result_alpha_tensor)


class PreviewImageAlpha:
    """Preview images with alpha transparency (no background replacement)."""

    CATEGORY = "Video/Masking"
    RETURN_TYPES = ()
    FUNCTION = "preview_alpha"
    OUTPUT_NODE = True

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


class SaveImageSequenceZip:
    """Save connected image/mask sequences as files in a ZIP archive."""

    CATEGORY = "Video/Masking"
    RETURN_TYPES = ()
    FUNCTION = "save_sequence"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
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

    def save_sequence(self, **kwargs):
        """Save connected image/mask sequences in a ZIP file."""

        inputs = []
        for index in range(1, 9):
            data = kwargs.get(f"input{index}")
            if data is None:
                continue
            prefix = kwargs.get(f"input{index}_prefix", f"input{index}")
            prefix = prefix.strip() or f"input{index}"
            inputs.append((f"input{index}", data, prefix))

        # Generate random suffix for zip file to avoid overwriting
        zip_suffix = random.randint(0, 9999999)
        zip_filename = f"save_to_zip_{zip_suffix:07d}.zip"

        # Get output directory
        output_dir = folder_paths.get_output_directory()
        zip_path = os.path.join(output_dir, zip_filename)

        # Create ZIP file
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for input_name, data, prefix in inputs:
                if data is None:
                    continue
                if isinstance(data, str):
                    payload = data.encode("utf-8")
                    ext = self._extension_for_text(data)
                    file_name = f"{prefix}.{ext}"
                    zipf.writestr(file_name, payload)
                    continue
                if isinstance(data, dict):
                    payload = json.dumps(data, indent=2).encode("utf-8")
                    file_name = f"{prefix}.json"
                    zipf.writestr(file_name, payload)
                    continue
                frames = self._normalize_frames(data)
                if frames.shape[0] == 1:
                    pil_img = self._to_pil(frames[0], "png")
                    img_buffer = BytesIO()
                    pil_img.save(img_buffer, format="PNG")
                    img_buffer.seek(0)
                    image_filename = f"{prefix}.png"
                    zipf.writestr(image_filename, img_buffer.getvalue())
                else:
                    digits = max(2, len(str(frames.shape[0])))
                    for i in range(frames.shape[0]):
                        pil_img = self._to_pil(frames[i], "png")
                        img_buffer = BytesIO()
                        pil_img.save(img_buffer, format="PNG")
                        img_buffer.seek(0)
                        image_filename = f"{prefix}_{i + 1:0{digits}d}.png"
                        zipf.writestr(image_filename, img_buffer.getvalue())

        _log(f"Saved ZIP to {zip_path}")

        # Create download URL (ComfyUI's /view endpoint)
        download_url = f"/view?filename={zip_filename}&type=output"

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
    def _normalize_frames(data: torch.Tensor) -> np.ndarray:
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
