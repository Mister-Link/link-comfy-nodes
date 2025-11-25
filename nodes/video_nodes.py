from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

import folder_paths

try:
    from aiohttp import web

    from server import PromptServer
except Exception:  # pragma: no cover - ComfyUI runtime handles availability
    PromptServer = None
    web = None

VIDEO_EXTENSIONS = ["webm", "mp4", "mkv", "gif", "mov"]
BIGMAX = 2**53 - 1
DIMMAX = 8192
DEFAULT_PREVIEW_FRAME_LIMIT = 48


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


def _clear_stale_masks():
    """Clear mask regions to prevent stale data."""
    _mask_regions.clear()
    _mask_regions_by_video.clear()
    _mask_versions.clear()
    _log("Cleared all mask regions")


def _increment_mask_version(node_id: str) -> int:
    """Track mask updates to invalidate execution cache."""
    if not node_id:
        return 0
    node_id = str(node_id)
    _mask_versions[node_id] = _mask_versions.get(node_id, 0) + 1
    _log(f"Mask version for node {node_id} -> {_mask_versions[node_id]}")
    return _mask_versions[node_id]


def _get_mask_version(node_id: Optional[str]) -> int:
    if not node_id:
        return 0
    return _mask_versions.get(str(node_id), 0)


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
    """Load a video, create masks for each frame, and expose preview endpoints."""

    CATEGORY = "Video/Masking"
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("frames", "masks", "frame_count")
    FUNCTION = "load_video"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.split(".")[-1].lower() in VIDEO_EXTENSIONS
        ]

        return {
            "required": {
                "video": (sorted(files),),
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
                "mask_crops_frames": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"force_size": "STRING", "unique_id": "UNIQUE_ID"},
        }

    def load_video(
        self,
        video: str,
        framerate: int,
        custom_width: int,
        custom_height: int,
        frame_load_cap: int,
        skip_first_frames: int,
        select_every_nth: int,
        mask_crops_frames: bool,
        force_size: str = "",
        unique_id: Optional[str] = None,
    ):
        video_path = folder_paths.get_annotated_filepath(video)
        processing_result = _load_video_frames(
            video_path,
            framerate,
            custom_width,
            custom_height,
            frame_load_cap,
            skip_first_frames,
            select_every_nth,
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

        frames_tensor = torch.from_numpy(frames_array)

        _log(f"mask_crops_frames setting: {mask_crops_frames}")
        _log(f"unique_id: {unique_id}")
        _log(f"Available mask regions: {list(_mask_regions.keys())}")
        _log(f"_mask_regions dict id: {id(_mask_regions)}")

        masks = torch.zeros(
            (frames_tensor.shape[0], target_height, target_width), dtype=torch.float32
        )
        region = None
        if unique_id and str(unique_id) in _mask_regions:
            region = _mask_regions[str(unique_id)]
            _log(f"Found region for unique_id {unique_id}: {region}")
        else:
            _log(f"No mask region found for unique_id: {unique_id}")

        if region:
            region = region.clamp(target_width, target_height)
            if region.width and region.height:
                masks[
                    :,
                    region.y : region.y + region.height,
                    region.x : region.x + region.width,
                ] = 1.0
                _log(
                    f"Applied mask region: x={region.x}, y={region.y}, w={region.width}, h={region.height}"
                )

                # Crop frames if enabled
                if mask_crops_frames:
                    _log(f"Original frames shape: {frames_tensor.shape}")
                    frames_tensor = frames_tensor[
                        :,
                        region.y : region.y + region.height,
                        region.x : region.x + region.width,
                        :,
                    ]
                    # When cropping, create new masks that are all white (the entire output is the selected region)
                    masks = torch.ones(
                        (frames_tensor.shape[0], region.height, region.width),
                        dtype=torch.float32,
                    )
                    _log(
                        f"Cropped frames to mask region. New shape: {frames_tensor.shape}"
                    )
            else:
                _log(
                    f"Region has zero width or height: w={region.width}, h={region.height}"
                )

        output_dir = folder_paths.get_output_directory()
        for idx, frame_data in enumerate(frames_list):
            frame_255 = (frame_data * 255).astype(np.uint8)
            pil_img = Image.fromarray(frame_255)
            preview_name = f"vme_frame_{unique_id}_{idx:04d}.png"
            pil_img.save(os.path.join(output_dir, preview_name))

        _log(
            f"Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}"
        )
        json.dumps(selected_frame_indices)  # backward compatibility noop

        return frames_tensor, masks, frames_tensor.shape[0]

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        video_path = folder_paths.get_annotated_filepath(video)
        mask_version = _get_mask_version(kwargs.get("unique_id"))
        if os.path.exists(video_path):
            return f"{os.path.getmtime(video_path)}_{mask_version}"
        return f"missing_{mask_version}"

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
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
        max_preview_frames = min(
            _int_param("max_preview_frames", DEFAULT_PREVIEW_FRAME_LIMIT, 1), 120
        )

        try:
            processing_result = _load_video_frames(
                folder_paths.get_annotated_filepath(video_name),
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

        frames_payload = []
        for idx, frame_data in enumerate(processing_result["frames"]):
            frame_255 = np.clip(frame_data * 255.0, 0, 255).astype(np.uint8)
            if frame_255.ndim == 3 and frame_255.shape[2] == 1:
                frame_255 = frame_255[:, :, 0]

            pil_img = Image.fromarray(frame_255)
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            frames_payload.append(
                {
                    "index": idx,
                    "data": encoded,
                    "width": pil_img.width,
                    "height": pil_img.height,
                }
            )

        return web.json_response(
            {
                "frames": frames_payload,
                "fps": processing_result["effective_fps"],
                "original_fps": processing_result["original_fps"],
                "selected_frame_indices": processing_result["selected_indices"],
                "frame_count": processing_result["total_frames"],
            }
        )


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


_register_preview_route()
_register_mask_route()
_register_clear_mask_route()


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
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10000,
                        "step": 1,
                        "display": "number",
                    },
                ),
            }
        }

    def calculate_wan_frames(self, frame_count: int):
        if frame_count <= 1:
            return (1,)
        wan_frames = 1 + (int(np.ceil((frame_count - 1) / 4)) * 4)
        _log(f"Input frames: {frame_count} → WAN frames: {wan_frames}")
        return (wan_frames,)
