"""Video detailer — refines a video latent with drift-aware compositing."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

import comfy.samplers
import nodes
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion


class VideoDetailer:
    """
    Details a video latent with drift-aware pixel-space compositing.

    Pipeline:
      1. Decode original latent → reference frames (or use explicit reference_image)
      2. KSampler denoise (with noise_mask + DifferentialDiffusion + boosted VACE)
      3. Decode denoised latent → enhanced frames
      4. Build a drift map: per-pixel distance between enhanced and reference
      5. Where drift exceeds threshold, blend back toward reference
      6. Apply mask compositing on top: unmasked areas always come from reference

    This prevents the model from silently changing eye color, accessories, etc.
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
                "vace_strength_mult": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Multiplier for VACE reference strength. "
                        ">1 = stronger reference adherence during denoising.",
                    },
                ),
                "drift_threshold": (
                    "FLOAT",
                    {
                        "default": 0.12,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Per-pixel color distance threshold (0-1). "
                        "Pixels drifting more than this from the reference "
                        "get blended back. 0 = snap everything to reference, "
                        "1 = allow all changes. Try 0.08-0.15.",
                    },
                ),
                "drift_blend": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "How much to pull drifted pixels back toward reference. "
                        "1.0 = fully replace with reference, "
                        "0.0 = keep enhanced. Try 0.5-0.8.",
                    },
                ),
            },
            "optional": {
                "mask_opt": ("MASK",),
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Refines a video latent with drift-aware compositing. "
        "Compares denoised output to a reference image and pulls back "
        "pixels that drifted too far (eye color changes, added accessories, etc). "
        "Unmasked areas are always preserved from the reference."
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

    @staticmethod
    def _scale_vace_strength(conditioning, mult):
        """Deep-copy conditioning and multiply all vace_strength values."""
        if mult == 1.0:
            return conditioning
        out = []
        for tensor, meta in conditioning:
            meta = copy.copy(meta)
            if "vace_strength" in meta:
                meta["vace_strength"] = [s * mult for s in meta["vace_strength"]]
            out.append([tensor, meta])
        return out

    @staticmethod
    def _build_drift_correction(
        enhanced: torch.Tensor,
        reference: torch.Tensor,
        threshold: float,
        blend: float,
    ) -> torch.Tensor:
        """Blend enhanced frames back toward reference where drift exceeds threshold.

        Args:
            enhanced:  [F, H, W, C] enhanced pixel frames
            reference: [F, H, W, C] reference pixel frames
            threshold: max per-pixel color distance before correction kicks in (0-1)
            blend:     how much to pull back toward reference (0=keep enhanced, 1=full ref)

        Returns:
            [F, H, W, C] corrected frames
        """
        if blend <= 0.0 or threshold >= 1.0:
            return enhanced

        # Per-pixel color distance across channels: [F, H, W]
        diff = (enhanced - reference).abs().mean(dim=-1)

        # Smooth the drift map to avoid pixel-level noise
        # Use the gaussian blur on each frame
        kernel_size = 5
        sigma = 1.5
        x = (
            torch.arange(kernel_size, dtype=torch.float32, device=diff.device)
            - kernel_size // 2
        )
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        k1d = gauss / gauss.sum()
        k2d = (k1d.unsqueeze(0) * k1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        pad = kernel_size // 2

        smoothed_frames = []
        for frame_diff in diff:
            smoothed = (
                F.conv2d(frame_diff.unsqueeze(0).unsqueeze(0), k2d, padding=pad)
                .squeeze(0)
                .squeeze(0)
            )
            smoothed_frames.append(smoothed)
        diff_smooth = torch.stack(smoothed_frames)

        # Build correction weight: 0 where diff <= threshold, ramps up above
        # Soft ramp over a band of 0.5*threshold width above threshold
        ramp_width = max(threshold * 0.5, 0.01)
        correction_weight = ((diff_smooth - threshold) / ramp_width).clamp(0.0, 1.0)
        # Scale by blend strength
        correction_weight = correction_weight * blend
        # [F, H, W] → [F, H, W, 1]
        correction_weight = correction_weight.unsqueeze(-1)

        corrected = (1 - correction_weight) * enhanced + correction_weight * reference
        return corrected

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
        noise_mask_feather,
        vace_strength_mult,
        drift_threshold,
        drift_blend,
        mask_opt=None,
        reference_image=None,
    ):
        model, clip, vae, positive, negative = basic_pipe

        # WAN latent: [B, C, F, H, W]
        latent_samples = latent["samples"]
        img_height = latent_samples.shape[3] * 8
        img_width = latent_samples.shape[4] * 8

        print(
            f"[Video Detailer] latent {latent_samples.shape} → "
            f"{img_width}×{img_height} px"
        )

        # --- Step 1: Get reference frames ---
        # WAN VAE temporally upscales: pixel_frames = max(0, latent_frames*4 - 3)
        original_decoded = vae.decode(latent_samples)
        original_frames = self._fix_decoded_shape(original_decoded, img_height)
        num_frames = original_frames.shape[0]
        print(
            f"[Video Detailer] decoded frames {original_frames.shape} ({num_frames} frames)"
        )

        # Use explicit reference_image if provided, otherwise use decoded latent
        if reference_image is not None:
            ref_frames = reference_image
            # Expand single image to all frames
            if ref_frames.shape[0] == 1:
                ref_frames = ref_frames.expand(num_frames, -1, -1, -1)
            elif ref_frames.shape[0] != num_frames:
                ref_frames = ref_frames[0:1].expand(num_frames, -1, -1, -1)
            # Resize if spatial dims don't match
            if (
                ref_frames.shape[1] != original_frames.shape[1]
                or ref_frames.shape[2] != original_frames.shape[2]
            ):
                ref_frames = ref_frames.permute(0, 3, 1, 2)  # [F,H,W,C] → [F,C,H,W]
                ref_frames = F.interpolate(
                    ref_frames,
                    size=(original_frames.shape[1], original_frames.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                )
                ref_frames = ref_frames.permute(0, 2, 3, 1)  # [F,C,H,W] → [F,H,W,C]
            print(f"[Video Detailer] using explicit reference image {ref_frames.shape}")
        else:
            ref_frames = original_frames
            print("[Video Detailer] using decoded latent as reference")

        # --- Step 2: Build pixel-space mask [F, H, W] ---
        if mask_opt is None:
            mask = torch.ones((num_frames, img_height, img_width), dtype=torch.float32)
        else:
            mask = mask_opt
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(num_frames, -1, -1)
            elif mask.shape[0] != num_frames:
                mask = mask[0:1].expand(num_frames, -1, -1)

        # --- Step 3: Build latent noise_mask (feathered) for KSampler ---
        feathered_latent = self._gaussian_blur_mask(mask, noise_mask_feather)
        noise_mask = feathered_latent.unsqueeze(1).to(latent_samples.device)

        print(
            f"[Video Detailer] noise_mask {noise_mask.shape} "
            f"mean={noise_mask.mean():.3f}"
        )

        # DifferentialDiffusion makes the model only denoise pixels whose mask
        # value >= the current noise threshold at each step.
        if "denoise_mask_function" not in model.model_options:
            model = DifferentialDiffusion.execute(model)[0]
            print("[Video Detailer] DifferentialDiffusion applied")

        # --- Step 4: Boost VACE reference strength for this detailer pass ---
        boosted_positive = self._scale_vace_strength(positive, vace_strength_mult)
        boosted_negative = self._scale_vace_strength(negative, vace_strength_mult)
        if vace_strength_mult != 1.0:
            print(f"[Video Detailer] VACE strength ×{vace_strength_mult}")

        # --- Step 5: KSampler denoising ---
        latent_dict = {"samples": latent_samples, "noise_mask": noise_mask}

        samples = nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            boosted_positive,
            boosted_negative,
            latent_dict,
            denoise=denoise,
        )[0]

        # --- Step 6: Decode the DENOISED latent ---
        enhanced_decoded = vae.decode(samples["samples"])
        enhanced_frames = self._fix_decoded_shape(enhanced_decoded, img_height)
        print(f"[Video Detailer] enhanced frames {enhanced_frames.shape}")

        # --- Step 7: Drift correction ---
        # Compare enhanced to reference; pull back pixels that drifted too far
        corrected = self._build_drift_correction(
            enhanced_frames.to(ref_frames.device),
            ref_frames,
            drift_threshold,
            drift_blend,
        )
        print(
            f"[Video Detailer] drift correction applied "
            f"(threshold={drift_threshold}, blend={drift_blend})"
        )

        # --- Step 8: Mask compositing ---
        # Unmasked areas always come from reference, masked areas get corrected result
        composite_mask = self._gaussian_blur_mask(mask, noise_mask_feather)
        # [F, H, W] → [F, H, W, 1] for broadcasting with [F, H, W, C]
        mask_4d = composite_mask.unsqueeze(-1).to(ref_frames.device)

        output = (1 - mask_4d) * ref_frames + mask_4d * corrected
        print(f"[Video Detailer] composited {output.shape}")

        output_mask = mask[0] if mask.ndim == 3 else mask
        return (output, output_mask)


__all__ = ["VideoDetailer"]
