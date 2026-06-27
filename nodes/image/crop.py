from __future__ import annotations

import torch


class CropToContentNode:
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "alpha")
    FUNCTION = "crop_to_content"
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "alpha": ("MASK",),
            },
        }

    def crop_to_content(self, images, alpha=None):
        frames = images.detach().cpu().float()
        if frames.ndim != 4:
            raise ValueError("Expected images with shape (N, H, W, C)")

        if alpha is not None:
            mask = alpha.detach().cpu().float()
            if mask.ndim == 4 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.ndim != 3:
                raise ValueError("Expected alpha mask with shape (N, H, W)")
            if mask.shape[0] != frames.shape[0]:
                raise ValueError("Alpha mask batch size does not match images")
            if mask.shape[1] != frames.shape[1] or mask.shape[2] != frames.shape[2]:
                raise ValueError("Alpha mask dimensions do not match images")

            mask_max = float(mask.max().item()) if mask.numel() else 0.0
            if mask_max > 1.0:
                mask = mask / 255.0

            mask = mask.clamp(0, 1)
        else:
            mask = torch.ones(
                frames.shape[0],
                frames.shape[1],
                frames.shape[2],
                device=frames.device,
                dtype=frames.dtype,
            )

        threshold = 0.01
        is_content = mask <= threshold

        if not is_content.any():
            frames = frames[:, :1, :1, :]
            mask = mask[:, :1, :1]
            return (frames, mask)

        coords = is_content.nonzero(as_tuple=False)

        height = frames.shape[1]
        width = frames.shape[2]

        y_min = int(coords[:, 1].min().item())
        x_min = int(coords[:, 2].min().item())
        y_max = int(coords[:, 1].max().item()) + 1
        x_max = int(coords[:, 2].max().item()) + 1

        y_min = max(y_min - 1, 0)
        x_min = max(x_min - 1, 0)
        y_max = min(y_max + 1, height)
        x_max = min(x_max + 1, width)

        frames = frames[:, y_min:y_max, x_min:x_max, :]
        mask = mask[:, y_min:y_max, x_min:x_max]

        return (frames, mask)
