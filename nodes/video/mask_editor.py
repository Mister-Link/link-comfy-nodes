from __future__ import annotations

import base64
import json
import os
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol, TypedDict, cast

import cv2
import numpy as np
import torch
from PIL import Image

import folder_paths  # type: ignore[import-untyped]

from ...utils import parse_hex_color
from .frame_loading import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    BIGMAX,
    FrameLoadResult,
    _coerce_int,
    load_frames_from_folder,
    load_video_frames,
)

try:
    from aiohttp import web

    from server import PromptServer  # type: ignore[import-not-found]
except Exception:
    PromptServer = None
    web = None

DEFAULT_PREVIEW_FRAME_LIMIT = 120
PREVIEW_CACHE_MAX_ITEMS = 10
PREVIEW_CACHE_MAX_BYTES = 30_000_000


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


@dataclass
class MaskRegion:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, object] | None) -> MaskRegion | None:
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

    def clamp(self, max_width: int, max_height: int) -> MaskRegion:
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

_KEYFRAMES_CACHE_FILE = os.path.join(
    folder_paths.get_output_directory(), "video_mask_keyframes.json"
)


def _save_keyframes_to_disk():
    try:
        with open(_KEYFRAMES_CACHE_FILE, "w") as f:
            json.dump(_mask_keyframes, f)
    except Exception:
        pass


def _load_keyframes_from_disk():
    global _mask_keyframes
    try:
        if os.path.exists(_KEYFRAMES_CACHE_FILE):
            with open(_KEYFRAMES_CACHE_FILE, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    _mask_keyframes = cast(dict[str, dict[str, MaskKeyframe]], loaded)
                else:
                    _mask_keyframes = {}
    except Exception:
        pass


def _increment_mask_version(node_id: str) -> int:
    if not node_id:
        return 0
    node_id = str(node_id)
    _mask_versions[node_id] = _mask_versions.get(node_id, 0) + 1
    _preview_cache.clear()
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
        return

    _preview_cache[key] = payload
    _preview_cache.move_to_end(key)

    while len(_preview_cache) > PREVIEW_CACHE_MAX_ITEMS:
        _preview_cache.popitem(last=False)


def _generate_masks_from_keyframes(
    node_id: str | None, frame_count: int, width: int, height: int
) -> np.ndarray:
    if not node_id or str(node_id) not in _mask_keyframes:
        return np.zeros((frame_count, height, width), dtype=np.float32)

    keyframes = _mask_keyframes[str(node_id)]
    if not keyframes:
        return np.zeros((frame_count, height, width), dtype=np.float32)

    sorted_keyframes = sorted(keyframes.items(), key=lambda x: int(x[0]))
    keyframe_indices = [int(kf[0]) for kf in sorted_keyframes]

    masks = np.zeros((frame_count, height, width), dtype=np.float32)

    for frame_idx in range(frame_count):
        if frame_idx < keyframe_indices[0]:
            continue

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
            continue

        mask_type = prev_kf_data.get("type", "bbox")

        if mask_type == "empty":
            continue

        if mask_type in ("bbox", "hybrid") and prev_kf_data.get("bbox"):
            prev_bbox = prev_kf_data.get("bbox", {})

            if (
                next_kf_data
                and next_kf_data.get("bbox")
                and next_kf_idx is not None
                and prev_kf_idx is not None
            ):
                next_bbox = next_kf_data.get("bbox", {})
                total_frames = next_kf_idx - prev_kf_idx
                if total_frames > 0:
                    t = (frame_idx - prev_kf_idx) / total_frames
                else:
                    t = 0.0

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
                x = int(prev_bbox.get("x", 0))
                y = int(prev_bbox.get("y", 0))
                w = int(prev_bbox.get("width", width))
                h = int(prev_bbox.get("height", height))

            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(0, min(w, width - x))
            h = max(0, min(h, height - y))

            masks[frame_idx, y : y + h, x : x + w] = 1.0

        if mask_type in ("painted", "hybrid") and prev_kf_data.get("mask_data"):
            mask_data = prev_kf_data.get("mask_data", "")
            if mask_data:
                try:
                    decoded = base64.b64decode(mask_data)
                    mask_array = np.frombuffer(decoded, dtype=np.float32)

                    stored_width = prev_kf_data.get("mask_width", width)
                    stored_height = prev_kf_data.get("mask_height", height)

                    expected_size = stored_height * stored_width
                    if len(mask_array) != expected_size:
                        stored_height = int(np.sqrt(len(mask_array)))
                        stored_width = len(mask_array) // stored_height

                    mask_array = mask_array.reshape((stored_height, stored_width))

                    if stored_width != width or stored_height != height:
                        mask_array = cv2.resize(
                            mask_array,
                            (width, height),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    masks[frame_idx] = np.maximum(masks[frame_idx], mask_array)
                except Exception:
                    pass

    return masks


class VideoMaskEditor:
    CATEGORY: str = "Video/Masking"
    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "INT", "MASK", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("frames", "frame_count", "masks", "alpha_channel")
    FUNCTION: str = "load_video"
    OUTPUT_NODE: bool = False

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        items = []

        for f in os.listdir(input_dir):
            full_path = os.path.join(input_dir, f)
            if (
                os.path.isfile(full_path)
                and f.split(".")[-1].lower() in VIDEO_EXTENSIONS
            ):
                items.append(f)
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
        _ = force_size

        video_path = folder_paths.get_annotated_filepath(source)

        if is_wan:
            source_total_frames = 0
            if os.path.isdir(video_path):
                image_files = [
                    f
                    for f in os.listdir(video_path)
                    if f.split(".")[-1].lower() in IMAGE_EXTENSIONS
                ]
                source_total_frames = len(image_files)
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

            combined_step = max(1, select_every_nth) * frame_step
            available_frames = max(0, source_total_frames - skip_first_frames)
            available_sampled = (available_frames + combined_step - 1) // combined_step

            if frame_load_cap == 0:
                target_frames = available_sampled
            else:
                target_frames = min(frame_load_cap, available_sampled)

            n = max(1, (target_frames - 1) // 4)
            snapped_cap = 4 * n + 1
            if snapped_cap < 5:
                snapped_cap = 5

            if frame_load_cap != snapped_cap:
                frame_load_cap = snapped_cap

        if os.path.isdir(video_path):
            image_files = [
                f
                for f in os.listdir(video_path)
                if f.split(".")[-1].lower() in IMAGE_EXTENSIONS
            ]
            if not image_files:
                raise ValueError(f"Folder contains no image files: {source}")

            processing_result = load_frames_from_folder(
                video_path, framerate, 0, 0, frame_load_cap,
                skip_first_frames, select_every_nth,
            )
        elif os.path.isfile(video_path):
            ext = source.split(".")[-1].lower()
            if ext not in VIDEO_EXTENSIONS:
                raise ValueError(f"Not a valid video file: {source}")

            processing_result = load_video_frames(
                video_path, framerate, 0, 0, frame_load_cap,
                skip_first_frames, select_every_nth,
            )
        else:
            raise ValueError(
                f"Input is neither a valid video file nor a folder: {source}"
            )

        frames_list = processing_result["frames"]
        target_width = processing_result["target_width"]
        target_height = processing_result["target_height"]

        frames_array = np.array(frames_list, dtype=np.float32)
        if frames_array.ndim == 3:
            frames_array = np.expand_dims(frames_array, axis=3)
        frames_array = np.clip(frames_array, 0.0, 1.0)

        if frames_array.shape[-1] == 4:
            rgb_array = frames_array[:, :, :, :3]
            alpha_array = frames_array[:, :, :, 3]
        else:
            rgb_array = frames_array
            if rgb_array.shape[-1] == 1:
                rgb_array = np.repeat(rgb_array, 3, axis=3)
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

        region = None
        if unique_id and str(unique_id) in _mask_regions:
            region = _mask_regions[str(unique_id)]

        if region:
            region = region.clamp(target_width, target_height)

        output_dir = folder_paths.get_output_directory()
        for idx in range(rgb_array.shape[0]):
            frame_rgb = (rgb_array[idx] * 255).astype(np.uint8)
            pil_img = Image.fromarray(frame_rgb)
            preview_name = f"vme_frame_{unique_id}_{idx:04d}.png"
            pil_img.save(
                os.path.join(output_dir, preview_name), compress_level=1, optimize=False
            )

        masks_array = _generate_masks_from_keyframes(
            unique_id, frames_tensor.shape[0], target_width, target_height
        )

        masks_tensor = torch.from_numpy(masks_array)

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


def _register_preview_route():
    if PromptServer is None or web is None:
        return

    server = PromptServer
    aiohttp_web = web

    @server.instance.routes.get("/videomaskeditor/preview")
    async def video_mask_editor_preview(request: _Request):
        params = request.rel_url.query
        video_name = params.get("video")
        node_id = params.get("node_id")

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
                video_path, video_mtime, framerate, custom_width, custom_height,
                frame_load_cap, skip_first_frames, select_every_nth,
                max_preview_frames, skip_mask, 0 if skip_mask else mask_version,
            )

            if cache_key in _preview_cache:
                cached_payload = _preview_cache[cache_key]
                _preview_cache.move_to_end(cache_key)
                return aiohttp_web.json_response(cached_payload)

            if os.path.isdir(video_path):
                processing_result = load_frames_from_folder(
                    video_path, framerate, custom_width, custom_height,
                    frame_load_cap, skip_first_frames, select_every_nth,
                    preview_max_frames=max_preview_frames,
                )
            else:
                processing_result = load_video_frames(
                    video_path, framerate, custom_width, custom_height,
                    frame_load_cap, skip_first_frames, select_every_nth,
                    preview_max_frames=max_preview_frames,
                )
        except Exception as exc:
            return aiohttp_web.json_response({"error": str(exc)}, status=400)

        masks_array = None
        if node_id and not skip_mask:
            try:
                masks_array = _generate_masks_from_keyframes(
                    node_id,
                    len(processing_result["frames"]),
                    processing_result["target_width"],
                    processing_result["target_height"],
                )
            except Exception:
                pass

        frames_payload: list[dict[str, object]] = []
        for idx, frame_data in enumerate(processing_result["frames"]):
            has_alpha = frame_data.shape[-1] == 4
            frame_255 = np.clip(frame_data * 255.0, 0, 255).astype(np.uint8)

            if masks_array is not None and idx < len(masks_array):
                mask = masks_array[idx]
                if np.any(mask > 0.5):
                    mask_3d = np.expand_dims(mask, axis=-1)

                    if has_alpha and frame_255.shape[-1] == 4:
                        alpha_channel = frame_255[:, :, 3].astype(np.float32)
                        frame_float = frame_255[:, :, :3].astype(np.float32)
                        red_mix = 0.5
                        red = np.array([255.0, 0.0, 0.0], dtype=np.float32)

                        frame_float = np.where(
                            mask_3d > 0.5,
                            np.clip(
                                frame_float + (red - frame_float) * red_mix, 0, 255
                            ),
                            frame_float,
                        )

                        alpha_channel = np.where(
                            (mask[:, :] > 0.5) & (alpha_channel < 128),
                            np.maximum(alpha_channel, 128),
                            alpha_channel,
                        )

                        frame_255[:, :, :3] = np.clip(frame_float, 0, 255).astype(
                            np.uint8
                        )
                        frame_255[:, :, 3] = np.clip(alpha_channel, 0, 255).astype(
                            np.uint8
                        )
                    else:
                        red_overlay = np.zeros_like(frame_255)
                        red_overlay[:, :, 0] = 255

                        alpha = 0.5
                        frame_255 = np.where(
                            mask_3d > 0.5,
                            cv2.addWeighted(
                                frame_255, 1 - alpha, red_overlay, alpha, 0
                            ),
                            frame_255,
                        ).astype(np.uint8)

            if frame_255.ndim == 3 and frame_255.shape[2] == 1:
                frame_255 = frame_255[:, :, 0]

            if has_alpha and frame_255.shape[-1] == 4:
                pil_img = Image.fromarray(frame_255, mode="RGBA")
            else:
                pil_img = Image.fromarray(frame_255)

            buffer = BytesIO()
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
    async def video_mask_editor_setmask(request: _Request):
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

            _increment_mask_version(node_id_str)

            server.instance.send_sync(
                "videomaskeditor.mask_updated", {"node_id": node_id_str}
            )

            return aiohttp_web.json_response({"success": True})
        except Exception as exc:
            return aiohttp_web.json_response({"error": str(exc)}, status=500)


def _register_clear_mask_route():
    if PromptServer is None or web is None:
        return

    server = PromptServer
    aiohttp_web = web

    @server.instance.routes.post("/videomaskeditor/clearmask")
    async def video_mask_editor_clearmask(request: _Request):
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
            _increment_mask_version(node_id_str)

            return aiohttp_web.json_response({"success": True})
        except Exception as exc:
            return aiohttp_web.json_response({"error": str(exc)}, status=500)


def _register_keyframe_routes():
    if PromptServer is None or web is None:
        return

    server = PromptServer
    aiohttp_web = web

    @server.instance.routes.post("/videomaskeditor/setkeyframe")
    async def video_mask_editor_setkeyframe(request: _Request):
        try:
            data = await request.json()
            node_id = data.get("node_id")
            frame_index = data.get("frame_index")
            mask_type = data.get("type", "bbox")
            bbox_data = data.get("bbox")
            mask_data = data.get("mask_data")

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

            _mask_keyframes[node_id_str][str(frame_index)] = mask_entry

            _increment_mask_version(node_id_str)
            _save_keyframes_to_disk()

            server.instance.send_sync(
                "videomaskeditor.mask_updated", {"node_id": node_id_str}
            )

            return aiohttp_web.json_response({"success": True})
        except Exception as exc:
            return aiohttp_web.json_response({"error": str(exc)}, status=500)

    @server.instance.routes.post("/videomaskeditor/deletekeyframe")
    async def video_mask_editor_deletekeyframe(request: _Request):
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
                _increment_mask_version(node_id_str)
                _save_keyframes_to_disk()

                server.instance.send_sync(
                    "videomaskeditor.mask_updated", {"node_id": node_id_str}
                )

            return aiohttp_web.json_response({"success": True})
        except Exception as exc:
            return aiohttp_web.json_response({"error": str(exc)}, status=500)

    @server.instance.routes.get("/videomaskeditor/getkeyframes")
    async def video_mask_editor_getkeyframes(request: _Request):
        try:
            params = request.rel_url.query
            node_id = params.get("node_id")

            if not node_id:
                return aiohttp_web.json_response(
                    {"error": "Missing node_id"}, status=400
                )

            keyframes = _mask_keyframes.get(str(node_id), {})
            return aiohttp_web.json_response({"keyframes": keyframes})
        except Exception as exc:
            return aiohttp_web.json_response({"error": str(exc)}, status=500)

    @server.instance.routes.get("/videomaskeditor/calculate_wan_cap")
    async def video_mask_editor_calculate_wan_cap(request: _Request):
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

            if os.path.isdir(video_path):
                image_files = [
                    f
                    for f in os.listdir(video_path)
                    if f.split(".")[-1].lower() in IMAGE_EXTENSIONS
                ]
                source_total_frames = len(image_files)
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

            combined_step = max(1, select_every_nth) * frame_step
            available_frames = max(0, source_total_frames - skip_first_frames)
            available_sampled = (available_frames + combined_step - 1) // combined_step

            if available_sampled < 5:
                wan_cap = -1
            else:
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
            return aiohttp_web.json_response({"error": str(exc)}, status=500)

    @server.instance.routes.post("/videomaskeditor/restorekeyframes")
    async def video_mask_editor_restorekeyframes(request: _Request):
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

            _mask_keyframes[node_id_str] = keyframes

            _increment_mask_version(node_id_str)
            _save_keyframes_to_disk()

            server.instance.send_sync(
                "videomaskeditor.mask_updated", {"node_id": node_id_str}
            )

            return aiohttp_web.json_response({"success": True})
        except Exception as exc:
            return aiohttp_web.json_response({"error": str(exc)}, status=500)


_register_preview_route()
_register_mask_route()
_register_clear_mask_route()
_register_keyframe_routes()
_load_keyframes_from_disk()
