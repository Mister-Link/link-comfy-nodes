from __future__ import annotations

import os
import tempfile
import uuid

import imageio.v3 as iio
import numpy as np
import torch

try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None  # type: ignore


class PreviewWebmNode:
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_webm"
    CATEGORY = "image/preview"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.1, "max": 120.0, "step": 0.1},
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def _empty_ui():
        return {"ui": {"webm_preview": []}}

    @staticmethod
    def _to_video_array(frames: torch.Tensor) -> tuple[np.ndarray, int]:
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4:
            raise ValueError(f"expected 4D tensor, got shape={tuple(frames.shape)}")

        frame_count = int(frames.shape[0])
        if frame_count == 0:
            raise ValueError("empty frame batch")

        np_frames = frames.detach().cpu().numpy()
        np_frames = np.clip(np_frames, 0.0, 1.0)

        channels = int(np_frames.shape[-1])
        if channels == 1:
            np_frames = np.repeat(np_frames, 3, axis=-1)
        elif channels >= 3:
            np_frames = np_frames[..., :3]
        else:
            raise ValueError(f"unsupported channel count: {channels}")

        np_frames = np.ascontiguousarray((np_frames * 255.0).round().astype(np.uint8))

        if np_frames.shape[0] == 1:
            np_frames = np.concatenate([np_frames, np_frames], axis=0)

        return np_frames, frame_count

    @staticmethod
    def _encode_webm(filepath: str, frames_uint8: np.ndarray, fps: float):
        strategies = (
            {
                "codec": "libvpx-vp9",
                "pixelformat": "yuv420p",
                "output_params": [
                    "-crf", "10", "-b:v", "0",
                    "-deadline", "good", "-cpu-used", "4",
                ],
            },
            {
                "codec": "libvpx",
                "pixelformat": "yuv420p",
                "output_params": ["-deadline", "good", "-cpu-used", "4"],
            },
            {},
        )

        errors: list[str] = []
        for options in strategies:
            try:
                iio.imwrite(filepath, frames_uint8, fps=fps, **options)
                if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
                    return
                errors.append(f"empty output file with options={options!r}")
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc} options={options!r}")
            finally:
                if os.path.isfile(filepath) and os.path.getsize(filepath) == 0:
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass

        raise RuntimeError(" ; ".join(errors))

    def preview_webm(self, frames: torch.Tensor, fps: float):
        print(
            f"[webm_preview] called, frames type={type(frames)}, "
            f"shape={getattr(frames, 'shape', None)}, fps={fps}"
        )
        if not isinstance(frames, torch.Tensor):
            return self._empty_ui()

        try:
            video_frames, frame_count = self._to_video_array(frames)
        except Exception as exc:
            print(f"[webm_preview] invalid input: {exc}")
            return self._empty_ui()

        if folder_paths:
            temp_dir = folder_paths.get_temp_directory()
        else:
            temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)

        unique_id = uuid.uuid4().hex[:8]
        filename = f"webm_preview_{unique_id}.webm"
        filepath = os.path.join(temp_dir, filename)

        print(f"[webm_preview] encoding {video_frames.shape[0]} frames to {filepath}")

        try:
            self._encode_webm(filepath, video_frames, float(fps))
            print(f"[webm_preview] encoded OK, size={os.path.getsize(filepath)}")
        except Exception as exc:
            print(f"[webm_preview] encode FAILED: {exc}")
            return self._empty_ui()

        return {
            "ui": {
                "webm_preview": [
                    {
                        "filename": filename,
                        "subfolder": "",
                        "type": "temp",
                        "frame_count": frame_count,
                        "fps": fps,
                    }
                ]
            }
        }
