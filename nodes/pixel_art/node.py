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

    @staticmethod
    def _pick_processing_device(frames: torch.Tensor) -> torch.device:
        # Prefer the incoming tensor device. If inputs are on CPU, opportunistically
        # use a hardware accelerator for the heavy convolution work.
        device = frames.device
        if device.type == "cpu":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
        return device

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
        alpha: torch.Tensor | None = None,
    ):
        process_device = self._pick_processing_device(frames)
        images = frames.detach().to(device=process_device, dtype=torch.float32)
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
            mask = alpha.detach().to(device=process_device, dtype=torch.float32)
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

        pixelated = torch.stack(outputs).clamp(0, 1).cpu()
        alpha_mask = torch.stack(alpha_outputs).clamp(0, 1).cpu()

        return (pixelated, alpha_mask)
