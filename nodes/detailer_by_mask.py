"""Video detailer — refines a video latent with pixel-space compositing."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.samplers
import nodes
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion


class VideoDetailer:
    """
    Refines a video latent using KSampler + noise mask, then composites
    the result back onto the original decoded frames in pixel space.

    The noise_mask + DifferentialDiffusion controls where denoising happens.
    Pixel-space compositing guarantees unmasked areas are pixel-identical
    to the original. An optional reference_image replaces the decoded
    original as the compositing base for unmasked regions.
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
                "feather": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Gaussian blur radius for the pixel-space "
                        "paste-back mask edge.",
                    },
                ),
                "noise_mask_feather": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Gaussian blur radius for the latent noise mask "
                        "(DifferentialDiffusion boundary softness).",
                    },
                ),
            },
            "optional": {
                "mask_opt": ("MASK",),
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": "If provided, unmasked areas in the output come "
                        "from this image instead of the decoded latent."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Refines a video latent with pixel-space compositing. "
        "Denoised result is blended onto original frames using the mask. "
        "Unmasked areas are never modified."
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
        self,
        latent,
        basic_pipe,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        feather,
        noise_mask_feather,
        mask_opt=None,
        reference_image=None,
    ):
        model, clip, vae, positive, negative = basic_pipe

        # WAN latent: [B, C, F, H, W]
        latent_samples = latent["samples"]
        img_height = latent_samples.shape[3] * 8
        img_width = latent_samples.shape[4] * 8

        print(
            f"[Video Detailer] latent {latent_samples.shape} → {img_width}×{img_height} px"
        )

        # --- Step 1: Decode the ORIGINAL latent → pixel frames ---
        # WAN VAE temporally upscales: pixel_frames = max(0, latent_frames*4 - 3)
        original_decoded = vae.decode(latent_samples)
        original_frames = self._fix_decoded_shape(original_decoded, img_height)
        num_frames = original_frames.shape[0]
        print(f"[Video Detailer] decoded {num_frames} frames {original_frames.shape}")

        # --- Step 2: Build pixel-space mask [F, H, W] ---
        if mask_opt is None:
            mask = torch.ones((num_frames, img_height, img_width), dtype=torch.float32)
        else:
            mask = mask_opt.clone()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(num_frames, -1, -1).contiguous()
            elif mask.shape[0] != num_frames:
                mask = mask[0:1].expand(num_frames, -1, -1).contiguous()

        # --- Step 3: Build latent noise_mask for KSampler ---
        feathered_latent = self._gaussian_blur_mask(mask, noise_mask_feather)
        noise_mask = feathered_latent.unsqueeze(1).to(latent_samples.device)
        # [F_pixel, 1, H, W] — ComfyUI's reshape_mask will resample to latent dims

        print(
            f"[Video Detailer] noise_mask {noise_mask.shape} mean={noise_mask.mean():.3f}"
        )

        # DifferentialDiffusion: denoise strength varies spatially per mask value
        if "denoise_mask_function" not in model.model_options:
            model = DifferentialDiffusion.execute(model)[0]
            print("[Video Detailer] DifferentialDiffusion applied")

        # --- Step 4: KSampler denoising ---
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

        # --- Step 5: Decode the DENOISED latent ---
        enhanced_decoded = vae.decode(samples["samples"])
        enhanced_frames = self._fix_decoded_shape(enhanced_decoded, img_height)
        print(f"[Video Detailer] enhanced {enhanced_frames.shape}")

        # --- Step 6: Choose the base for unmasked areas ---
        if reference_image is not None:
            base_frames = reference_image
            if base_frames.shape[0] == 1:
                base_frames = base_frames.expand(num_frames, -1, -1, -1)
            elif base_frames.shape[0] != num_frames:
                base_frames = base_frames[0:1].expand(num_frames, -1, -1, -1)
            # Resize if needed
            if base_frames.shape[1] != img_height or base_frames.shape[2] != img_width:
                bf = base_frames.permute(0, 3, 1, 2)
                bf = F.interpolate(
                    bf,
                    size=(img_height, img_width),
                    mode="bilinear",
                    align_corners=False,
                )
                base_frames = bf.permute(0, 2, 3, 1)
            base_frames = base_frames.contiguous()
            print(f"[Video Detailer] using reference_image as base {base_frames.shape}")
        else:
            base_frames = original_frames
            print("[Video Detailer] using decoded original as base")

        # --- Step 7: Pixel-space compositing ---
        # Feather the mask for smooth blending at paste edges
        composite_mask = self._gaussian_blur_mask(mask, feather)
        mask_4d = composite_mask.unsqueeze(-1).to(base_frames.device)

        # Match frame counts
        out_frames = min(base_frames.shape[0], enhanced_frames.shape[0])
        output = (1 - mask_4d[:out_frames]) * base_frames[:out_frames] + mask_4d[
            :out_frames
        ] * enhanced_frames[:out_frames].to(base_frames.device)
        print(f"[Video Detailer] output {output.shape}")

        output_mask = mask[0] if mask.ndim == 3 else mask
        return (output, output_mask)


__all__ = ["VideoDetailer"]
