from __future__ import annotations

import json
import math

import numpy as np
import torch


class SpritesheetBuilderNode:
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("spritesheet", "alpha", "metadata")
    FUNCTION = "build_spritesheet"
    CATEGORY = "image/transform"

    _ASPECT_RATIOS = {
        "1:1 (Square)": (1, 1),
        "4:3 (Landscape)": (4, 3),
        "3:4 (Portrait)": (3, 4),
        "16:9 (Landscape)": (16, 9),
        "9:16 (Portrait)": (9, 16),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "alpha": ("MASK",),
                "aspect_ratio": (list(cls._ASPECT_RATIOS.keys()),),
            }
        }

    def build_spritesheet(self, frames, alpha=None, aspect_ratio: str = "1:1 (Square)"):
        frames_cpu = frames.detach().cpu().float()
        if frames_cpu.ndim != 4:
            raise ValueError("Expected frames with shape (N, H, W, C)")

        alpha_cpu = None
        if alpha is not None:
            alpha_tensor = alpha.detach().cpu().float()
            if alpha_tensor.ndim == 4 and alpha_tensor.shape[-1] == 1:
                alpha_tensor = alpha_tensor[..., 0]
            if alpha_tensor.ndim != 3:
                raise ValueError("Expected alpha mask with shape (N, H, W)")
            if alpha_tensor.shape[0] != frames_cpu.shape[0]:
                raise ValueError("Alpha mask batch size does not match frames")
            alpha_cpu = alpha_tensor.clamp(0, 1)

        target_ratio = self._aspect_ratio_value(aspect_ratio)
        frame_count, frame_height, frame_width, frame_channels = frames_cpu.shape
        use_alpha = alpha_cpu is not None or frame_channels == 4

        if use_alpha and frame_channels == 3:
            alpha_stack = (
                alpha_cpu
                if alpha_cpu is not None
                else torch.ones(
                    (frame_count, frame_height, frame_width),
                    dtype=frames_cpu.dtype,
                )
            )
            frames_cpu = torch.cat((frames_cpu, alpha_stack.unsqueeze(-1)), dim=-1)
            frame_channels = 4
        elif not use_alpha and frame_channels == 4:
            frames_cpu = frames_cpu[..., :3]
            frame_channels = 3

        columns, rows = self._closest_grid(
            frame_count, target_ratio, frame_width, frame_height
        )
        sheet_width = frame_width * columns
        sheet_height = frame_height * rows

        spritesheet = np.zeros(
            (sheet_height, sheet_width, frame_channels), dtype=np.float32
        )

        frames_np = frames_cpu.numpy()
        for idx in range(frame_count):
            row = idx // columns
            col = idx % columns
            y0 = row * frame_height
            x0 = col * frame_width
            spritesheet[
                y0 : y0 + frame_height, x0 : x0 + frame_width, :frame_channels
            ] = frames_np[idx, :, :, :frame_channels]

        result = torch.from_numpy(spritesheet).unsqueeze(0)
        if frame_channels == 4:
            alpha_mask = result[..., 3].clone()
        else:
            alpha_mask = torch.ones((1, sheet_height, sheet_width), dtype=result.dtype)
        metadata = {
            "spritesheet": {
                "width": sheet_width,
                "height": sheet_height,
                "rows": rows,
                "columns": columns,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "frame_count": frame_count,
            }
        }
        return (result, alpha_mask, json.dumps(metadata, indent=2))

    @classmethod
    def _aspect_ratio_value(cls, aspect_ratio: str) -> float:
        width, height = cls._ASPECT_RATIOS[aspect_ratio]
        return width / height

    @staticmethod
    def _closest_grid(
        frame_count: int, target_ratio: float, frame_width: int, frame_height: int
    ) -> tuple[int, int]:
        best_cols = 1
        best_rows = frame_count
        best_diff = float("inf")

        for cols in range(1, frame_count + 1):
            rows = math.ceil(frame_count / cols)
            ratio = (cols * frame_width) / (rows * frame_height)
            diff = abs(ratio - target_ratio)

            if diff < best_diff:
                best_diff = diff
                best_cols = cols
                best_rows = rows

        return best_cols, best_rows
