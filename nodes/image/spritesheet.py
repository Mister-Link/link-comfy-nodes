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
    # Building one spritesheet requires every frame at once, even when they
    # arrive as a list of variable-sized images (e.g. from Load Folder), so
    # this must run as a single call rather than being mapped per list item.
    INPUT_IS_LIST = True

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

    @staticmethod
    def _flatten_frames(value) -> list[torch.Tensor]:
        if isinstance(value, (list, tuple)):
            frames: list[torch.Tensor] = []
            for item in value:
                frames.extend(SpritesheetBuilderNode._flatten_frames(item))
            return frames
        tensor = value.detach().cpu().float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError("Expected frames with shape (N, H, W, C)")
        return [tensor[i] for i in range(tensor.shape[0])]

    @staticmethod
    def _flatten_masks(value) -> list[torch.Tensor]:
        if isinstance(value, (list, tuple)):
            masks: list[torch.Tensor] = []
            for item in value:
                masks.extend(SpritesheetBuilderNode._flatten_masks(item))
            return masks
        tensor = value.detach().cpu().float()
        if tensor.ndim == 4 and tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError("Expected alpha mask with shape (N, H, W)")
        return [tensor[i] for i in range(tensor.shape[0])]

    def build_spritesheet(self, frames, alpha=None, aspect_ratio="1:1 (Square)"):
        if isinstance(aspect_ratio, (list, tuple)):
            aspect_ratio = aspect_ratio[0]

        frame_list = self._flatten_frames(frames)
        if not frame_list:
            raise ValueError("Expected at least one frame")
        frame_count = len(frame_list)

        alpha_list = None
        if alpha is not None:
            alpha_list = self._flatten_masks(alpha)
            if len(alpha_list) != frame_count:
                raise ValueError("Alpha mask count does not match frame count")

        target_ratio = self._aspect_ratio_value(aspect_ratio)

        # Frames may not all share the same size, so the grid cell is sized to
        # fit the largest frame; smaller frames are placed in the top-left of
        # their cell rather than being resized or cropped to match the rest.
        frame_width = max(frame.shape[1] for frame in frame_list)
        frame_height = max(frame.shape[0] for frame in frame_list)
        use_alpha = alpha_list is not None or any(
            frame.shape[-1] == 4 for frame in frame_list
        )
        frame_channels = 4 if use_alpha else 3

        columns, rows = self._closest_grid(
            frame_count, target_ratio, frame_width, frame_height
        )
        sheet_width = frame_width * columns
        sheet_height = frame_height * rows

        spritesheet = np.zeros(
            (sheet_height, sheet_width, frame_channels), dtype=np.float32
        )

        for idx, frame in enumerate(frame_list):
            frame_np = frame.numpy()
            h, w = frame_np.shape[0], frame_np.shape[1]

            row = idx // columns
            col = idx % columns
            y0 = row * frame_height
            x0 = col * frame_width

            spritesheet[y0 : y0 + h, x0 : x0 + w, :3] = frame_np[:, :, :3]
            if frame_channels == 4:
                if alpha_list is not None:
                    spritesheet[y0 : y0 + h, x0 : x0 + w, 3] = (
                        alpha_list[idx].clamp(0, 1).numpy()
                    )
                elif frame_np.shape[-1] == 4:
                    spritesheet[y0 : y0 + h, x0 : x0 + w, 3] = frame_np[:, :, 3]
                else:
                    spritesheet[y0 : y0 + h, x0 : x0 + w, 3] = 1.0

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
