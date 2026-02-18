"""Video detailer — refines a video latent using a noise mask."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.samplers
import nodes
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion


class VideoDetailer:
    """
    Details a video latent by running ksampler with a noise mask.

    The latent is passed unchanged; the noise mask restricts denoising to the
    masked region. VACE conditioning (with reference image) is carried through
    basic_pipe and guides the output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "basic_pipe": ("BASIC_PIPE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0001, "max": 1.0, "step": 0.01},
                ),
                "noise_mask_feather": (
                    "INT",
                    {"default": 20, "min": 0, "max": 200, "step": 1},
                ),
            },
            "optional": {
                "mask_opt": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Refines a video latent using ksampler + noise mask. "
        "VACE reference image influence comes through basic_pipe conditioning."
    )

    @staticmethod
    def _gaussian_blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask

        kernel_size = radius * 2 + 1
        min_dim = min(mask.shape[-1], mask.shape[-2])
        if min_dim <= kernel_size:
            kernel_size = min_dim // 2
            if kernel_size % 2 == 0:
                kernel_size -= 1
            if kernel_size < 3:
                return mask

        sigma = kernel_size / 3.0
        x = (
            torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
            - kernel_size // 2
        )
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        k1d = gauss / gauss.sum()
        k2d = (k1d.unsqueeze(0) * k1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        pad = kernel_size // 2

        def _blur_one(m2d):
            return (
                F.conv2d(m2d.unsqueeze(0).unsqueeze(0), k2d, padding=pad)
                .squeeze(0)
                .squeeze(0)
            )

        if mask.ndim == 2:
            return _blur_one(mask)
        return torch.stack([_blur_one(m) for m in mask])

    @staticmethod
    def _fix_decoded_shape(decoded: torch.Tensor, expected_height: int) -> torch.Tensor:
        """Normalise WAN VAE output [B,F,d1,d2,C] → [F,H,W,C]."""
        if decoded.ndim == 5:
            b, f, d1, d2, c = decoded.shape
            decoded = decoded.reshape(b * f, d1, d2, c)
        if decoded.shape[1] != expected_height:
            decoded = torch.rot90(decoded, k=3, dims=(1, 2)).contiguous()
        return decoded

    def execute(
        dsaself,
        latent,
        basic_pipe,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        noise_mask_feather,
        mask_opt=None,
    ):
        model, clip, vae, positive, negative = basic_pipe

        # WAN latent: [B, C, F, H, W]
        latent_samples = latent["samples"]
        num_frames = latent_samples.shape[2]
        img_height = latent_samples.shape[3] * 8
        img_width = latent_samples.shape[4] * 8

        print(
            f"[Video Detailer] latent {latent_samples.shape} → "
            f"{img_width}×{img_height} px, {num_frames} frames"
        )

        # Build pixel-space mask [F, H, W]
        if mask_opt is None:
            mask = torch.ones((num_frames, img_height, img_width), dtype=torch.float32)
        else:
            mask = mask_opt
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(num_frames, -1, -1)
            elif mask.shape[0] != num_frames:
                mask = mask[0:1].expand(num_frames, -1, -1)

        # Feather and add channel dim → [F, 1, H, W] for ComfyUI reshape_mask
        feathered = self._gaussian_blur_mask(mask, noise_mask_feather)
        noise_mask = feathered.unsqueeze(1).to(latent_samples.device)

        print(
            f"[Video Detailer] noise_mask {noise_mask.shape} "
            f"mean={noise_mask.mean():.3f}"
        )

        # DifferentialDiffusion makes the model only denoise pixels whose mask
        # value >= the current noise threshold at each step. Without it, the
        # noise_mask only composites at the very end and the model drifts
        # everything during sampling regardless of the mask.
        if "denoise_mask_function" not in model.model_options:
            model = DifferentialDiffusion.execute(model)[0]
            print("[Video Detailer] DifferentialDiffusion applied")

        latent_dict = {"samples": latent_samples, "noise_mask": noise_mask}

        samples = nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_dict,
            denoise=denoise,
        )[0]

        decoded = vae.decode(samples["samples"])
        print(f"[Video Detailer] decoded {decoded.shape}")
        decoded = self._fix_decoded_shape(decoded, img_height)
        print(f"[Video Detailer] fixed   {decoded.shape}")

        output_mask = mask[0] if mask.ndim == 3 else mask
        return (decoded, output_mask)


__all__ = ["VideoDetailer"]
