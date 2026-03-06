from __future__ import annotations

import torch

from ...utils import parse_hex_color


class ReplaceAlpha:
    CATEGORY: str = "Video/Masking"
    RETURN_TYPES: tuple[str, ...] = ("IMAGE", "MASK")
    RETURN_NAMES: tuple[str, ...] = ("frames", "alpha")
    FUNCTION: str = "replace_alpha"
    OUTPUT_NODE: bool = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "alpha": ("MASK",),
                "mask": ("MASK",),
                "color": ("STRING", {"default": "#FFFFFF"}),
            }
        }

    def replace_alpha(
        self,
        frames: torch.Tensor,
        alpha: torch.Tensor,
        mask: torch.Tensor,
        color: str,
    ):
        alpha_tensor = alpha
        mask_tensor = mask
        if alpha_tensor.ndim == 4 and alpha_tensor.shape[-1] == 1:
            alpha_tensor = alpha_tensor[..., 0]
        if mask_tensor.ndim == 4 and mask_tensor.shape[-1] == 1:
            mask_tensor = mask_tensor[..., 0]

        if (
            frames.shape[0] != alpha_tensor.shape[0]
            or frames.shape[0] != mask_tensor.shape[0]
        ):
            raise ValueError(
                f"Frame count mismatch: frames={frames.shape[0]}, alpha={alpha_tensor.shape[0]}, mask={mask_tensor.shape[0]}"
            )

        if (
            frames.shape[1:3] != alpha_tensor.shape[1:3]
            or frames.shape[1:3] != mask_tensor.shape[1:3]
        ):
            raise ValueError(
                f"Frame size mismatch: frames={frames.shape[1:3]}, alpha={alpha_tensor.shape[1:3]}, mask={mask_tensor.shape[1:3]}"
            )

        r, g, b = parse_hex_color(color, fallback=(255, 255, 255))
        color_rgb = torch.tensor(
            [r / 255.0, g / 255.0, b / 255.0],
            device=frames.device,
            dtype=frames.dtype,
        )

        result_alpha = alpha_tensor.clone()
        result_frames = frames.clone()
        # Only replace pixels where alpha is transparent AND mask covers the area
        replace_regions = (alpha_tensor < 0.5) & (mask_tensor > 0.5)
        if replace_regions.any():
            replace_regions_3d = replace_regions.unsqueeze(-1).expand_as(result_frames)
            result_frames = torch.where(replace_regions_3d, color_rgb.expand_as(result_frames), result_frames)
            result_alpha = torch.where(
                replace_regions, torch.ones_like(result_alpha), result_alpha
            )

        return (result_frames, result_alpha)
