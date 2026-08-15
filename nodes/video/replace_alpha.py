from __future__ import annotations

import torch

from ...utils import parse_hex_color


class ReplaceAlpha:
    CATEGORY: str = "Video/Masking"
    RETURN_TYPES: tuple[str, ...] = ("IMAGE",)
    RETURN_NAMES: tuple[str, ...] = ("frames",)
    FUNCTION: str = "replace_alpha"
    OUTPUT_NODE: bool = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "White = replace with color, black = keep the "
                            "original frame content untouched."
                        )
                    },
                ),
                "color": ("STRING", {"default": "#FFFFFF"}),
            }
        }

    def replace_alpha(
        self,
        frames: torch.Tensor,
        mask: torch.Tensor,
        color: str,
    ):
        mask_tensor = mask
        if mask_tensor.ndim == 4 and mask_tensor.shape[-1] == 1:
            mask_tensor = mask_tensor[..., 0]

        if frames.shape[0] != mask_tensor.shape[0]:
            raise ValueError(
                f"Frame count mismatch: frames={frames.shape[0]}, mask={mask_tensor.shape[0]}"
            )
        if frames.shape[1:3] != mask_tensor.shape[1:3]:
            raise ValueError(
                f"Frame size mismatch: frames={frames.shape[1:3]}, mask={mask_tensor.shape[1:3]}"
            )

        r, g, b = parse_hex_color(color, fallback=(255, 255, 255))
        color_rgb = torch.tensor(
            [r / 255.0, g / 255.0, b / 255.0],
            device=frames.device,
            dtype=frames.dtype,
        )

        rgb = frames[..., :3]
        replace_regions = (mask_tensor > 0.5).unsqueeze(-1)
        color_broadcast = color_rgb.view(1, 1, 1, 3).expand_as(rgb)
        new_rgb = torch.where(replace_regions, color_broadcast, rgb)

        # Embed the result as an alpha-aware RGBA frame instead of a separate
        # MASK output: color-filled (replaced) regions become transparent,
        # preserved content stays opaque -- matches the embedded-alpha
        # convention the rest of this node family (e.g. Match Colors to
        # Reference) already reads automatically.
        alpha = (1.0 - mask_tensor).clamp(0.0, 1.0)
        result_frames = torch.cat([new_rgb, alpha.unsqueeze(-1)], dim=-1)

        return (result_frames,)
