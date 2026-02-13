from __future__ import annotations

import torch

from .pixel_effect import PixelEffectModule


class ConvertToPixelArt:
    """Convert input frames into pixel art while preserving transparency."""

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("pixelated_frames", "alpha")
    FUNCTION = "convert"
    CATEGORY = "image/transform"

    _model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "kernel_size": (
                    "INT",
                    {"default": 9, "min": 1, "max": 128, "step": 1},
                ),
                "pixel_size": (
                    "INT",
                    {"default": 11, "min": 1, "max": 128, "step": 1},
                ),
                "num_bins": (
                    "INT",
                    {"default": 10, "min": 1, "max": 256, "step": 1},
                ),
                "alpha_threshold": (
                    "FLOAT",
                    {"default": 0.58, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "clean_stray_pixels": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "stray_pixel_guard": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "alpha": ("MASK",),
            },
        }

    @classmethod
    def _get_model(cls) -> PixelEffectModule:
        if cls._model is None:
            cls._model = PixelEffectModule()
            cls._model.eval()
        return cls._model

    def convert(
        self,
        frames: torch.Tensor,
        kernel_size: int,
        pixel_size: int,
        num_bins: int,
        alpha_threshold: float,
        clean_stray_pixels: bool,
        stray_pixel_guard: float,
        alpha: torch.Tensor | None = None,
    ):
        images = frames.detach().cpu().float()
        if images.ndim != 4:
            raise ValueError("Expected frames with shape (N, H, W, C)")
        if images.numel():
            max_val = float(images.max())
            if max_val > 2.0:
                images = images / 255.0
            images = images.clamp(0.0, 1.0)

        has_alpha = images.shape[-1] == 4
        rgb = images[..., :3] * 255.0

        if alpha is not None:
            mask = alpha.detach().cpu().float()
            if mask.ndim == 4 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.ndim != 3:
                raise ValueError("Expected alpha mask with shape (N, H, W)")
            if mask.shape[0] != images.shape[0]:
                raise ValueError("Alpha mask batch size does not match frames")
            if mask.numel():
                max_val = float(mask.max())
                if max_val > 2.0:
                    mask = mask / 255.0
                mask = mask.clamp(0.0, 1.0)
            # Treat mask as alpha (white = opaque).
            alpha_channel = mask * 255.0
        elif has_alpha:
            alpha_channel = images[..., 3] * 255.0
        else:
            alpha_channel = (
                torch.ones(
                    images.shape[0],
                    images.shape[1],
                    images.shape[2],
                    device=images.device,
                    dtype=images.dtype,
                )
                * 255.0
            )

        model = self._get_model()
        outputs = []
        alpha_outputs = []

        # Plain-language controls:
        # - clean_stray_pixels: turn tiny wrong-color speck cleanup on/off.
        # - stray_pixel_guard: higher = more aggressive cleanup.
        guard = float(max(0.0, min(1.0, stray_pixel_guard)))
        if clean_stray_pixels:
            dominance_threshold = 0.62 + (0.20 * guard)
            outlier_filter = True
            outlier_color_delta_threshold = 90.0 - (34.0 * guard)
        else:
            dominance_threshold = 0.0
            outlier_filter = False
            outlier_color_delta_threshold = 72.0

        with torch.no_grad():
            for idx in range(images.shape[0]):
                rgb_pt = rgb[idx].permute(2, 0, 1).unsqueeze(0)
                alpha_pt = alpha_channel[idx].unsqueeze(0).unsqueeze(0)

                result_rgb_pt, result_alpha_pt = model(
                    rgb_pt,
                    alpha_pt,
                    param_num_bins=num_bins,
                    param_kernel_size=kernel_size,
                    param_pixel_size=pixel_size,
                    alpha_threshold=alpha_threshold,
                    dominance_threshold=dominance_threshold,
                    outlier_filter=outlier_filter,
                    outlier_color_delta_threshold=outlier_color_delta_threshold,
                )

                result_rgb = (
                    result_rgb_pt.squeeze(0).permute(1, 2, 0).clamp(0, 255) / 255.0
                )
                result_alpha = (
                    result_alpha_pt.squeeze(0).squeeze(0).clamp(0, 255) / 255.0
                )

                if has_alpha:
                    output = torch.cat([result_rgb, result_alpha.unsqueeze(-1)], dim=2)
                else:
                    output = result_rgb

                outputs.append(output)
                alpha_outputs.append(result_alpha)

        alpha_mask = torch.stack(alpha_outputs).clamp(0, 1)

        return (torch.stack(outputs), alpha_mask)
