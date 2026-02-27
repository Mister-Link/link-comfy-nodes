from __future__ import annotations

import os
from typing import TypedDict

import cv2
import numpy as np
from PIL import Image

VIDEO_EXTENSIONS = ["webm", "mp4", "mkv", "gif", "mov"]
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"]
BIGMAX = 2**53 - 1
DIMMAX = 8192


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str, bytes, bytearray)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


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


def load_frames_from_folder(
    folder_path: str,
    framerate: int,
    custom_width: int,
    custom_height: int,
    frame_load_cap: int,
    skip_first_frames: int,
    select_every_nth: int,
    preview_max_frames: int | None = None,
) -> FrameLoadResult:
    if not os.path.isdir(folder_path):
        raise ValueError(f"Not a directory: {folder_path}")

    image_files = []
    for fname in os.listdir(folder_path):
        if fname.split(".")[-1].lower() in IMAGE_EXTENSIONS:
            image_files.append(os.path.join(folder_path, fname))

    if not image_files:
        raise ValueError(f"No image files found in folder: {folder_path}")

    image_files.sort()

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

    assumed_fps = 18.0
    frame_step = (
        max(1, int(round(assumed_fps / framerate)))
        if framerate and framerate > 0
        else 1
    )
    combined_step = max(1, select_every_nth) * frame_step

    for frame_index, img_path in enumerate(image_files):
        if frame_index < skip_first_frames:
            continue

        relative_index = frame_index - skip_first_frames
        if relative_index % combined_step != 0:
            continue

        try:
            img = Image.open(img_path)
            if img.mode == "RGBA":
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

            frames_list.append(frame)
            selected_frame_indices.append(frame_index)
            img.close()

            if max_frames is not None and len(frames_list) >= max_frames:
                break
        except Exception:
            continue

    if not frames_list:
        raise RuntimeError("No frames loaded from folder")

    effective_fps = framerate if framerate > 0 else assumed_fps

    return {
        "frames": frames_list,
        "selected_indices": selected_frame_indices,
        "target_width": target_width,
        "target_height": target_height,
        "original_fps": int(assumed_fps),
        "effective_fps": effective_fps,
        "total_frames": len(image_files),
        "frame_step": frame_step,
        "combined_step": combined_step,
    }


def load_video_frames(
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
