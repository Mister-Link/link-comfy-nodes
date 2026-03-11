"""Temporal mask cropper - crops all frames to the union mask bounds across the batch."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TemporalMaskCropper:
    """Crop all frames to the union of mask bounds across the entire batch.

    Unlike per-frame cropping, this finds the bounding box that covers the mask
    in every frame, then applies that single crop region to all frames. This
    preserves temporal coherence so the subject stays at a consistent position.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "padding": ("INT", {"default": 16, "min": 0, "max": 512}),
                "use_image_padding": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "crop"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Crops all frames to the union bounding box of the mask across ALL frames. "
        "The same crop region is applied to every frame for temporal coherence. "
        "use_image_padding=True extends the crop window into surrounding image content "
        "(clamped at edges). use_image_padding=False fills padding with black."
    )

    def crop(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        padding: int = 16,
        use_image_padding: bool = True,
    ):
        """Crop all frames to the temporal union of mask bounds.

        Args:
            images: (T, H, W, C) batch of frames
            masks:  (T, H, W) or (1, H, W) batch of masks (values 0–1)
            padding: extra pixels around the union bounding box
            use_image_padding: if True, padding region uses real image pixels
                               (crop window is extended into the surrounding image,
                               clamped at frame edges). If False, padding is filled
                               with black (zeros).
        """
        T, H, W, _ = images.shape
        BM = masks.shape[0]

        # Resize masks to match image spatial dims if needed
        if masks.shape[1] != H or masks.shape[2] != W:
            masks = F.interpolate(
                masks.unsqueeze(1).float(), size=(H, W), mode="nearest-exact"
            ).squeeze(1)

        # Union of all non-zero mask pixels across the entire batch
        union_mask = masks.max(dim=0).values  # (H, W)

        y_idx, x_idx = torch.nonzero(union_mask > 0.5, as_tuple=True)

        if y_idx.numel() == 0:
            print("[TemporalMaskCropper] no mask content found, returning full frames")
            return (images, masks if BM == T else masks.expand(T, H, W))

        # Tight mask bounds
        tight_min_y = int(y_idx.min().item())
        tight_max_y = int(y_idx.max().item()) + 1
        tight_min_x = int(x_idx.min().item())
        tight_max_x = int(x_idx.max().item()) + 1

        if use_image_padding:
            # Extend crop window into surrounding image content, clamp at edges
            min_y = max(0, tight_min_y - padding)
            max_y = min(H, tight_max_y + padding)
            min_x = max(0, tight_min_x - padding)
            max_x = min(W, tight_max_x + padding)

            cropped_images = images[:, min_y:max_y, min_x:max_x, :]

            if BM == 1:
                cropped_masks = masks[0:1, min_y:max_y, min_x:max_x].expand(T, -1, -1)
            else:
                cropped_masks = masks[:, min_y:max_y, min_x:max_x]

            crop_h = max_y - min_y
            crop_w = max_x - min_x
        else:
            # Crop tight to mask, then pad with black
            cropped_images = images[:, tight_min_y:tight_max_y, tight_min_x:tight_max_x, :]
            # F.pad expects (C, H, W), pad order: left, right, top, bottom
            cropped_images = F.pad(
                cropped_images.permute(0, 3, 1, 2),
                (padding, padding, padding, padding),
                mode="constant",
                value=0,
            ).permute(0, 2, 3, 1)

            if BM == 1:
                tight_masks = masks[0:1, tight_min_y:tight_max_y, tight_min_x:tight_max_x]
            else:
                tight_masks = masks[:, tight_min_y:tight_max_y, tight_min_x:tight_max_x]

            cropped_masks = F.pad(
                tight_masks,
                (padding, padding, padding, padding),
                mode="constant",
                value=0,
            )
            if BM == 1:
                cropped_masks = cropped_masks.expand(T, -1, -1)

            crop_h = cropped_images.shape[1]
            crop_w = cropped_images.shape[2]

        print(
            f"[TemporalMaskCropper] → {crop_w}×{crop_h}  "
            f"(padding={padding}, use_image_padding={use_image_padding}, frames={T})"
        )

        return (cropped_images, cropped_masks)
