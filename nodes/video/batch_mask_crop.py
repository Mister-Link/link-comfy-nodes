"""Batch mask cropper that preserves mask movement across frames.

Instead of independently cropping each frame to its mask bounds (which makes
moving objects appear static), this tracks mask position and crops with a
consistent window that follows the movement trajectory.
"""

from __future__ import annotations

import numpy as np
import torch


class BatchMaskCropper:
    """Crop a batch of frames following mask movement.

    Tracks mask position across frames and applies consistent crop that
    preserves left-to-right, top-to-bottom, etc. motion.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "crop_size": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 512}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 512}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 512}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 512}),
                "extend_from_source": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Pad with original image edges where possible instead of black",
                    },
                ),
                "smooth": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Trajectory smoothing (0=no smooth, 1=max)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Crops a batch of frames following mask movement. Tracks mask position "
        "across frames and applies consistent crop window that preserves motion "
        "trajectory (e.g., circle moving left-to-right stays moving left-to-right). "
        "Supports directional padding with optional extension from source image."
    )

    def execute(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        crop_size: int,
        pad_left: int = 0,
        pad_top: int = 0,
        pad_right: int = 0,
        pad_bottom: int = 0,
        extend_from_source: bool = True,
        smooth: float = 0.5,
    ):
        """
        Args:
            images: (T, H, W, 3) batch of frames
            masks: (T, H, W) or (1, H, W) batch of masks
            crop_size: target crop window size
            pad_left/top/right/bottom: directional padding
            extend_from_source: use original image edges for padding
            smooth: trajectory smoothing factor
        """
        T, H, W = images.shape[0], images.shape[1], images.shape[2]

        # Expand mask to match image batch size if needed
        if masks.shape[0] == 1:
            masks = masks.expand(T, -1, -1)

        # Convert masks to numpy for processing
        masks_np = masks.cpu().numpy()  # (T, H, W)

        # Track mask centroid and bounds for each frame
        centers = []
        bounds = []

        for t in range(T):
            mask = masks_np[t]

            # Get non-zero mask coordinates
            coords = np.argwhere(mask > 0.5)  # (N, 2) in [y, x] format

            if len(coords) == 0:
                # No mask in this frame, use previous or center
                if centers:
                    centers.append(centers[-1])
                    bounds.append(bounds[-1])
                else:
                    centers.append(np.array([H / 2, W / 2]))
                    bounds.append(np.array([0, 0, H, W]))
            else:
                # Centroid
                center_y, center_x = coords.mean(axis=0)
                centers.append(np.array([center_y, center_x]))

                # Bounds
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                bounds.append(np.array([y_min, x_min, y_max, x_max]))

        centers = np.array(centers)  # (T, 2)
        bounds = np.array(bounds)  # (T, 4)

        # Smooth trajectory
        if smooth > 0.01:
            centers = self._smooth_trajectory(centers, smooth)

        # Compute crop window that accommodates all positions
        bound_centers_y = (bounds[:, 0] + bounds[:, 2]) / 2
        bound_centers_x = (bounds[:, 1] + bounds[:, 3]) / 2
        bound_sizes_h = bounds[:, 2] - bounds[:, 0] + pad_top + pad_bottom
        bound_sizes_w = bounds[:, 3] - bounds[:, 1] + pad_left + pad_right

        # Crop centers follow the mask centers
        crop_centers_y = bound_centers_y
        crop_centers_x = bound_centers_x

        # Ensure crop window size fits
        crop_h = max(crop_size, int(bound_sizes_h.max()) + pad_top + pad_bottom)
        crop_w = max(crop_size, int(bound_sizes_w.max()) + pad_left + pad_right)

        # Clamp to nearest valid size (square)
        max_crop = min(H, W)
        crop_size_final = min(max(crop_h, crop_w), max_crop)

        # Crop each frame following the trajectory
        cropped_images = []
        cropped_masks = []

        for t in range(T):
            cy = int(np.clip(crop_centers_y[t], crop_size_final // 2, H - crop_size_final // 2))
            cx = int(np.clip(crop_centers_x[t], crop_size_final // 2, W - crop_size_final // 2))

            y1 = cy - crop_size_final // 2
            y2 = y1 + crop_size_final
            x1 = cx - crop_size_final // 2
            x2 = x1 + crop_size_final

            # Clamp bounds and track how much padding is needed
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
                    crop_size_final,
                )
                mask_padded = self._pad_with_source(
                    masks[t:t+1].unsqueeze(-1),
                    mask_crop.unsqueeze(-1),
                    y1,
                    y2,
                    x1,
                    x2,
                    crop_size_final,
                ).squeeze(-1)
            else:
                img_padded = self._pad_to_size(img_crop, crop_size_final)
                mask_padded = self._pad_to_size(mask_crop, crop_size_final)

            cropped_images.append(img_padded)
            cropped_masks.append(mask_padded)

        result_images = torch.cat(cropped_images, dim=0)
        result_masks = torch.cat(cropped_masks, dim=0)

        print(f"[BatchMaskCropper] cropped {T} frames to {crop_size_final}x{crop_size_final}")

        return (result_images, result_masks)

    @staticmethod
    def _smooth_trajectory(trajectory: np.ndarray, factor: float) -> np.ndarray:
        """Smooth trajectory with exponential moving average."""
        smoothed = np.copy(trajectory)
        alpha = 1.0 - factor

        for t in range(1, len(trajectory)):
            smoothed[t] = alpha * trajectory[t] + (1 - alpha) * smoothed[t - 1]

        return smoothed

    @staticmethod
    def _pad_with_source(
        source: torch.Tensor,
        crop: torch.Tensor,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
        target_size: int,
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
                size=(target_size, target_size),
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
            size=(target_size, target_size),
            mode="nearest",
        ).permute(0, 2, 3, 1)

    @staticmethod
    def _pad_to_size(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        """Pad tensor to target size with black/zero padding, centered."""
        h, w = tensor.shape[1], tensor.shape[2]

        if h == target_size and w == target_size:
            return tensor

        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        tensor = torch.nn.functional.pad(
            tensor, (0, 0, pad_left, pad_right, pad_top, pad_bottom), value=0
        )

        return tensor
