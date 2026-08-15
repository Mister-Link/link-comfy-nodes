from __future__ import annotations

import json
import math

import numpy as np
import torch


class SpritesheetBuilderNode:
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("spritesheet", "frames", "metadata")
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

    def build_spritesheet(self, frames, aspect_ratio="1:1 (Square)"):
        if isinstance(aspect_ratio, (list, tuple)):
            aspect_ratio = aspect_ratio[0]

        frame_list = self._flatten_frames(frames)
        if not frame_list:
            raise ValueError("Expected at least one frame")
        frame_count = len(frame_list)

        target_ratio = self._aspect_ratio_value(aspect_ratio)

        # Frames may not all share the same size, so the grid cell is sized to
        # fit the largest frame; smaller frames are placed in the top-left of
        # their cell rather than being resized or cropped to match the rest.
        frame_width = max(frame.shape[1] for frame in frame_list)
        frame_height = max(frame.shape[0] for frame in frame_list)
        # Alpha convention: embedded 4th channel only. If any frame carries
        # alpha, the sheet gets an alpha channel (frames without one are
        # fully opaque; grid gaps stay transparent).
        use_alpha = any(frame.shape[-1] == 4 for frame in frame_list)
        frame_channels = 4 if use_alpha else 3

        columns, rows = self._closest_grid(
            frame_count, target_ratio, frame_width, frame_height
        )
        sheet_width = frame_width * columns
        sheet_height = frame_height * rows

        spritesheet = np.zeros(
            (sheet_height, sheet_width, frame_channels), dtype=np.float32
        )
        # Individual frames, normalized to one uniform batch: each frame
        # sits top-left in a cell-sized canvas exactly as it does in the
        # sheet (padding transparent when the sheet has alpha, black
        # otherwise).
        frames_out = np.zeros(
            (frame_count, frame_height, frame_width, frame_channels),
            dtype=np.float32,
        )

        for idx, frame in enumerate(frame_list):
            frame_np = frame.numpy()
            h, w = frame_np.shape[0], frame_np.shape[1]

            row = idx // columns
            col = idx % columns
            y0 = row * frame_height
            x0 = col * frame_width

            spritesheet[y0 : y0 + h, x0 : x0 + w, :3] = frame_np[:, :, :3]
            frames_out[idx, :h, :w, :3] = frame_np[:, :, :3]
            if frame_channels == 4:
                if frame_np.shape[-1] == 4:
                    cell_alpha = frame_np[:, :, 3]
                else:
                    cell_alpha = 1.0
                spritesheet[y0 : y0 + h, x0 : x0 + w, 3] = cell_alpha
                frames_out[idx, :h, :w, 3] = cell_alpha

        result = torch.from_numpy(spritesheet).unsqueeze(0)
        frames_tensor = torch.from_numpy(frames_out)
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
        return (result, frames_tensor, json.dumps(metadata, indent=2))

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
