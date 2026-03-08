"""Batch mask cropper - crops to mask bounds across a batch."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class BatchMaskCropper:
    """Crop batch of frames to mask bounds.

    Crops each frame to its mask bounds with optional padding.
    No background color fill - actual image content or edge cut-off.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 512}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "crop"
    CATEGORY = "link/video"
    DESCRIPTION = "Crops frames to mask bounds with optional padding. Returns actual image content, no background fill."

    def crop(self, images: torch.Tensor, masks: torch.Tensor, padding: int = 0):
        """Crop images to mask bounds.

        Args:
            images: (T, H, W, 3) batch of frames
            masks: (T, H, W) or (1, H, W) batch of masks
            padding: extra padding around mask bounds
        """
        B, H, W = images.shape[0], images.shape[1], images.shape[2]
        BM, HM, WM = masks.shape

        # Expand mask if needed
        if HM != H or WM != W:
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode="nearest-exact").squeeze(1)

        output_images = []
        output_masks = []

        # Crop each frame to its mask bounds
        for i in range(B):
            curr_mask = masks[i] if BM > 1 else masks[0]

            # Find bounds
            y_indices, x_indices = torch.nonzero(curr_mask > 0.5, as_tuple=True)

            if len(y_indices) == 0 or len(x_indices) == 0:
                continue

            # Get bounds with padding
            min_y = max(0, y_indices.min().item() - padding)
            max_y = min(H, y_indices.max().item() + 1 + padding)
            min_x = max(0, x_indices.min().item() - padding)
            max_x = min(W, x_indices.max().item() + 1 + padding)

            # Crop
            cropped_img = images[i, min_y:max_y, min_x:max_x, :]
            cropped_mask = curr_mask[min_y:max_y, min_x:max_x]

            output_images.append(cropped_img)
            output_masks.append(cropped_mask)

        if not output_images:
            return (torch.zeros((0, 1, 1, 3), dtype=images.dtype), torch.zeros((0, 1, 1), dtype=images.dtype))

        # Return as batch (may have different sizes, so return as list first)
        # Find max dimensions
        max_h = max(img.shape[0] for img in output_images)
        max_w = max(img.shape[1] for img in output_images)

        # Pad all to max size
        padded_images = []
        padded_masks = []

        for img, mask in zip(output_images, output_masks):
            h, w = img.shape[0], img.shape[1]
            pad_h = max_h - h
            pad_w = max_w - w

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # Pad images with edge extension
            padded_img = F.pad(img.permute(2, 0, 1), (pad_left, pad_right, pad_top, pad_bottom), mode="replicate").permute(1, 2, 0)
            padded_mask = F.pad(mask.unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0).squeeze(0)

            padded_images.append(padded_img)
            padded_masks.append(padded_mask)

        out_rgb = torch.stack(padded_images, dim=0)
        out_masks = torch.stack(padded_masks, dim=0)

        print(f"[BatchMaskCropper] cropped {len(output_images)} frames, max size {max_h}x{max_w}")

        return (out_rgb, out_masks)
