"""
Trim stabilizer padding based on strength parameters.

This node takes frames and masks from a video stabilizer and trims them
based on horizontal and vertical strength values. At 0.0 strength, no change
occurs. At 1.0 strength, the entire mask/padding on the respective axis is removed.
"""

from __future__ import annotations

import numpy as np
import torch


class StabilizerTrimNode:
    """Trim frames and masks from stabilized video based on padding strength."""

    CATEGORY: str = "Video/Stabilization"
    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("frames", "mask")
    FUNCTION: str = "trim_stabilized"
    OUTPUT_NODE: bool = False

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
        """
        Analyze a mask to determine padding on all sides.

        Returns (left_pad, right_pad, top_pad, bottom_pad) in pixels.
        The mask is expected to be 1.0 where padding exists, 0.0 where content is.
        """
        if mask.ndim == 3:
            # Take first frame if multiple frames
            mask_2d = mask[0]
        elif mask.ndim == 2:
            mask_2d = mask
        else:
            return 0, 0, 0, 0

        height, width = mask_2d.shape

        # Find where content exists (mask == 0)
        content_rows = np.any(mask_2d < 0.5, axis=1)
        content_cols = np.any(mask_2d < 0.5, axis=0)

        # Find first and last rows/cols with content
        rows_with_content = np.where(content_rows)[0]
        cols_with_content = np.where(content_cols)[0]

        if len(rows_with_content) == 0 or len(cols_with_content) == 0:
            # No content found, all padding
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
        """Crop a single frame and mask by removing pixels from each side."""
        height, width = frame.shape[:2]

        # Clamp crop values to valid range
        left = max(0, min(left, width - 1))
        right = max(0, min(right, width - left - 1))
        top = max(0, min(top, height - 1))
        bottom = max(0, min(bottom, height - top - 1))

        # Calculate crop region
        y_start = top
        y_end = height - bottom
        x_start = left
        x_end = width - right

        # Ensure we have at least 1x1 image
        if y_end <= y_start:
            y_start = 0
            y_end = height
        if x_end <= x_start:
            x_start = 0
            x_end = width

        # Crop frame (shape: H, W, C)
        cropped_frame = frame[y_start:y_end, x_start:x_end]

        # Crop mask (shape: H, W)
        cropped_mask = mask[y_start:y_end, x_start:x_end]

        return cropped_frame, cropped_mask

    def trim_stabilized(
        self,
        frames: torch.Tensor,
        mask: torch.Tensor,
        horizontal_strength: float,
        vertical_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trim frames and masks based on stabilization strength.

        Args:
            frames: Input frames tensor (N, H, W, C)
            mask: Input mask tensor (N, H, W) where 1.0 = padding
            horizontal_strength: 1.0 = keep all horizontal padding, 0.0 = remove all
            vertical_strength: 1.0 = keep all vertical padding, 0.0 = remove all

        Returns:
            Tuple of (trimmed_frames, trimmed_masks)
        """
        # Convert to numpy for processing
        frames_np = frames.cpu().numpy()
        mask_np = mask.cpu().numpy()

        # Use a single crop based on the union of content across all frames.
        # This avoids off-by-one differences that break stacking.
        mask_agg = np.min(mask_np, axis=0)
        left_pad, right_pad, top_pad, bottom_pad = self._analyze_mask_padding(mask_agg)

        # Calculate pixels to remove based on strength
        # Inverted: 1.0 = keep all padding, 0.0 = remove all padding
        # Proportional to the padding on each side
        left_crop = int(left_pad * (1.0 - horizontal_strength))
        right_crop = int(right_pad * (1.0 - horizontal_strength))
        top_crop = int(top_pad * (1.0 - vertical_strength))
        bottom_crop = int(bottom_pad * (1.0 - vertical_strength))

        # Process each frame and mask with the same crop
        num_frames = frames_np.shape[0]
        cropped_frames = []
        cropped_masks = []

        for i in range(num_frames):
            frame = frames_np[i]
            frame_mask = mask_np[i]

            cropped_frame, cropped_frame_mask = self._crop_frame_and_mask(
                frame, frame_mask, left_crop, right_crop, top_crop, bottom_crop
            )

            cropped_frames.append(cropped_frame)
            cropped_masks.append(cropped_frame_mask)

        # Stack back into tensors
        frames_out = torch.from_numpy(np.stack(cropped_frames, axis=0))
        masks_out = torch.from_numpy(np.stack(cropped_masks, axis=0))

        return frames_out, masks_out
