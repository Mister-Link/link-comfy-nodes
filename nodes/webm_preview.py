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

    def preview_webm(self, frames: torch.Tensor, fps: float):
        if not isinstance(frames, torch.Tensor) or frames.shape[0] == 0:
            return {"ui": {"webm_preview": []}}

        if folder_paths:
            temp_dir = folder_paths.get_temp_directory()
        else:
            temp_dir = tempfile.gettempdir()

        unique_id = uuid.uuid4().hex[:8]
        filename = f"webm_preview_{unique_id}.webm"
        filepath = os.path.join(temp_dir, filename)

        # frames shape: (N, H, W, C) float32 0-1 RGB
        frame_list = [(f.cpu().numpy() * 255).astype(np.uint8) for f in frames]

        iio.imwrite(
            filepath,
            frame_list,
            fps=fps,
            codec="libvpx-vp9",
            pixelformat="yuv420p",
            output_params=[
                "-crf",
                "10",
                "-b:v",
                "0",
                "-color_range",
                "2",
                "-deadline",
                "good",
                "-cpu-used",
                "4",
            ],
        )

        return {
            "ui": {
                "webm_preview": [
                    {
                        "filename": filename,
                        "subfolder": "",
                        "type": "temp",
                        "frame_count": frames.shape[0],
                        "fps": fps,
                    }
                ]
            }
        }
