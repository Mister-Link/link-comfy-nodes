from __future__ import annotations

import numpy as np
import torch


class StabilizerTrimNode:
    CATEGORY: str = "Video/Stabilization"
    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("frames", "mask")
    FUNCTION: str = "trim_stabilized"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "mask": ("MASK",),
                "horizontal_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "vertical_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            },
        }

    def _analyze_mask_padding(
        self, mask: np.ndarray
    ) -> tuple[int, int, int, int]:
        if mask.ndim == 3:
            mask_2d = mask[0]
        elif mask.ndim == 2:
            mask_2d = mask
        else:
            return 0, 0, 0, 0

        height, width = mask_2d.shape

        content_rows = np.any(mask_2d < 0.5, axis=1)
        content_cols = np.any(mask_2d < 0.5, axis=0)

        rows_with_content = np.where(content_rows)[0]
        cols_with_content = np.where(content_cols)[0]

        if len(rows_with_content) == 0 or len(cols_with_content) == 0:
            return 0, 0, 0, 0

        top_content = rows_with_content[0]
        bottom_content = rows_with_content[-1]
        left_content = cols_with_content[0]
        right_content = cols_with_content[-1]

        top_pad = top_content
        bottom_pad = height - bottom_content - 1
        left_pad = left_content
        right_pad = width - right_content - 1

        return left_pad, right_pad, top_pad, bottom_pad

    def _crop_frame_and_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        left: int,
        right: int,
        top: int,
        bottom: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = frame.shape[:2]

        left = max(0, min(left, width - 1))
        right = max(0, min(right, width - left - 1))
        top = max(0, min(top, height - 1))
        bottom = max(0, min(bottom, height - top - 1))

        y_start = top
        y_end = height - bottom
        x_start = left
        x_end = width - right

        if y_end <= y_start:
            y_start = 0
            y_end = height
        if x_end <= x_start:
            x_start = 0
            x_end = width

        cropped_frame = frame[y_start:y_end, x_start:x_end]
        cropped_mask = mask[y_start:y_end, x_start:x_end]

        return cropped_frame, cropped_mask

    def trim_stabilized(
        self,
        frames: torch.Tensor,
        mask: torch.Tensor,
        horizontal_strength: float,
        vertical_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames_np = frames.cpu().numpy()
        mask_np = mask.cpu().numpy()

        mask_agg = np.min(mask_np, axis=0)
        left_pad, right_pad, top_pad, bottom_pad = self._analyze_mask_padding(mask_agg)

        left_crop = int(left_pad * (1.0 - horizontal_strength))
        right_crop = int(right_pad * (1.0 - horizontal_strength))
        top_crop = int(top_pad * (1.0 - vertical_strength))
        bottom_crop = int(bottom_pad * (1.0 - vertical_strength))

        num_frames = frames_np.shape[0]
        cropped_frames = []
        cropped_masks = []

        for i in range(num_frames):
            cropped_frame, cropped_frame_mask = self._crop_frame_and_mask(
                frames_np[i], mask_np[i], left_crop, right_crop, top_crop, bottom_crop
            )
            cropped_frames.append(cropped_frame)
            cropped_masks.append(cropped_frame_mask)

        frames_out = torch.from_numpy(np.stack(cropped_frames, axis=0))
        masks_out = torch.from_numpy(np.stack(cropped_masks, axis=0))

        return frames_out, masks_out
