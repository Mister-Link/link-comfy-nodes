"""Batch mask cropper that preserves mask movement across frames.

Crops to a consistent window that contains all mask motion across the entire
batch, so moving objects actually move through the frame instead of jumping in place.
"""

from __future__ import annotations

import numpy as np
import torch


class BatchMaskCropper:
    """Crop a batch of frames to contain all mask movement.

    Calculates bounding box containing all mask positions across all frames,
    then crops consistently to that region so motion is preserved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "crop_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "crop_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 512}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 512}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 512}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 512}),
                "extend_from_source": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Pad with original image edges instead of black",
                    },
                ),
                "smooth_trajectory": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Smooth mask center across frames (off preserves raw motion)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Crops batch of frames to contain all mask movement. Instead of following "
        "the mask (which makes motion appear static), calculates the bounding box of "
        "all mask positions across all frames and crops to that region. Moving objects "
        "actually move through the frame. Supports directional padding with optional "
        "edge extension from source."
    )

    def execute(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        crop_width: int,
        crop_height: int,
        pad_left: int = 0,
        pad_top: int = 0,
        pad_right: int = 0,
        pad_bottom: int = 0,
        extend_from_source: bool = True,
        smooth_trajectory: bool = False,
    ):
        """
        Args:
            images: (T, H, W, 3) batch of frames
            masks: (T, H, W) or (1, H, W) batch of masks
            crop_width/height: target crop dimensions
            pad_left/top/right/bottom: directional padding
            extend_from_source: use original image edges for padding
            smooth_trajectory: smooth mask centers (default: off, preserves raw motion)
        """
        T, H, W = images.shape[0], images.shape[1], images.shape[2]

        # Expand mask to match image batch size if needed
        if masks.shape[0] == 1:
            masks = masks.expand(T, -1, -1)

        # Convert masks to numpy for processing
        masks_np = masks.cpu().numpy()  # (T, H, W)

        # Track mask bounds for each frame
        all_bounds = []

        for t in range(T):
            mask = masks_np[t]
            coords = np.argwhere(mask > 0.5)  # (N, 2) in [y, x] format

            if len(coords) == 0:
                # No mask in this frame, use image center
                all_bounds.append(np.array([H // 2, W // 2, H // 2, W // 2]))
            else:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                all_bounds.append(np.array([y_min, x_min, y_max, x_max]))

        all_bounds = np.array(all_bounds)  # (T, 4)

        # Calculate global bounding box containing ALL mask positions
        global_y_min = all_bounds[:, 0].min()
        global_x_min = all_bounds[:, 1].min()
        global_y_max = all_bounds[:, 2].max()
        global_x_max = all_bounds[:, 3].max()

        # Add padding
        global_y_min = max(0, int(global_y_min) - pad_top)
        global_x_min = max(0, int(global_x_min) - pad_left)
        global_y_max = min(H, int(global_y_max) + pad_bottom)
        global_x_max = min(W, int(global_x_max) + pad_right)

        # Calculate center of global bounds
        center_y = (global_y_min + global_y_max) / 2
        center_x = (global_x_min + global_x_max) / 2

        # Smooth if requested
        if smooth_trajectory:
            centers = np.array(
                [[(all_bounds[t, 0] + all_bounds[t, 2]) / 2, (all_bounds[t, 1] + all_bounds[t, 3]) / 2]
                 for t in range(T)]
            )
            centers = self._smooth_trajectory(centers)
            # Recalculate global bounds from smoothed centers
            center_y = centers[:, 0].mean()
            center_x = centers[:, 1].mean()

        # Crop each frame from the global bounds or center
        cropped_images = []
        cropped_masks = []

        for t in range(T):
            # Use global bounds for all frames (so motion is preserved)
            y1 = int(center_y - crop_height // 2)
            y2 = y1 + crop_height
            x1 = int(center_x - crop_width // 2)
            x2 = x1 + crop_width

            # Clamp to image bounds
            y1_clipped = max(0, y1)
            y2_clipped = min(H, y2)
            x1_clipped = max(0, x1)
            x2_clipped = min(W, x2)

            # Extract the valid region
            img_crop = images[t:t+1, y1_clipped:y2_clipped, x1_clipped:x2_clipped, :]
            mask_crop = masks[t:t+1, y1_clipped:y2_clipped, x1_clipped:x2_clipped]

            # Pad with extend_from_source or black padding
            if extend_from_source:
                img_padded = self._pad_with_source(
                    images[t:t+1],
                    img_crop,
                    y1,
                    y2,
                    x1,
                    x2,
                    crop_height,
                    crop_width,
                )
                mask_padded = self._pad_with_source(
                    masks[t:t+1].unsqueeze(-1),
                    mask_crop.unsqueeze(-1),
                    y1,
                    y2,
                    x1,
                    x2,
                    crop_height,
                    crop_width,
                ).squeeze(-1)
            else:
                img_padded = self._pad_to_size(img_crop, crop_height, crop_width)
                mask_padded = self._pad_to_size(mask_crop, crop_height, crop_width)

            cropped_images.append(img_padded)
            cropped_masks.append(mask_padded)

        result_images = torch.cat(cropped_images, dim=0)
        result_masks = torch.cat(cropped_masks, dim=0)

        print(
            f"[BatchMaskCropper] cropped {T} frames to {crop_width}x{crop_height} "
            f"(motion bounds: y[{global_y_min}:{global_y_max}] x[{global_x_min}:{global_x_max}])"
        )

        return (result_images, result_masks)

    @staticmethod
    def _smooth_trajectory(centers: np.ndarray) -> np.ndarray:
        """Smooth trajectory with exponential moving average."""
        smoothed = np.copy(centers)
        alpha = 0.3

        for t in range(1, len(centers)):
            smoothed[t] = alpha * centers[t] + (1 - alpha) * smoothed[t - 1]

        return smoothed

    @staticmethod
    def _pad_with_source(
        source: torch.Tensor,
        crop: torch.Tensor,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        """Pad crop with edges from source image where possible."""
        H, W = source.shape[1], source.shape[2]

        # Determine padding needed
        pad_top_needed = max(0, -y1)
        pad_bottom_needed = max(0, y2 - H)
        pad_left_needed = max(0, -x1)
        pad_right_needed = max(0, x2 - W)

        # If no padding needed, just resize
        if pad_top_needed == 0 and pad_bottom_needed == 0 and pad_left_needed == 0 and pad_right_needed == 0:
            return torch.nn.functional.interpolate(
                crop.permute(0, 3, 1, 2),
                size=(target_height, target_width),
                mode="nearest",
            ).permute(0, 2, 3, 1)

        # Extend from source edges
        if pad_top_needed > 0:
            top_edge = source[:, max(0, y1) : max(0, y1) + 1, max(0, x1) : min(W, x2), :]
            top_edge = top_edge.expand(-1, pad_top_needed, -1, -1)
            crop = torch.cat([top_edge, crop], dim=1)

        if pad_bottom_needed > 0:
            bottom_edge = source[:, min(H, y2) - 1 : min(H, y2), max(0, x1) : min(W, x2), :]
            bottom_edge = bottom_edge.expand(-1, pad_bottom_needed, -1, -1)
            crop = torch.cat([crop, bottom_edge], dim=1)

        if pad_left_needed > 0:
            left_edge = crop[:, :, :1, :].expand(-1, -1, pad_left_needed, -1)
            crop = torch.cat([left_edge, crop], dim=2)

        if pad_right_needed > 0:
            right_edge = crop[:, :, -1:, :].expand(-1, -1, pad_right_needed, -1)
            crop = torch.cat([crop, right_edge], dim=2)

        # Resize to target
        return torch.nn.functional.interpolate(
            crop.permute(0, 3, 1, 2),
            size=(target_height, target_width),
            mode="nearest",
        ).permute(0, 2, 3, 1)

    @staticmethod
    def _pad_to_size(tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Pad tensor to target size with black/zero padding, centered."""
        h, w = tensor.shape[1], tensor.shape[2]

        if h == target_height and w == target_width:
            return tensor

        pad_h = max(0, target_height - h)
        pad_w = max(0, target_width - w)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        tensor = torch.nn.functional.pad(
            tensor, (0, 0, pad_left, pad_right, pad_top, pad_bottom), value=0
        )

        return tensor
