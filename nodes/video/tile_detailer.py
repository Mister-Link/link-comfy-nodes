"""Video frame refiner — re-denoises a full video latent in one pass with an
optional reference image prepended as a protected temporal context frame.

The reference frame participates in temporal attention across all video frames,
acting as a consistent identity/style anchor throughout the sequence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.model_management  # type: ignore[import-untyped]
import comfy.samplers  # type: ignore[import-untyped]
import nodes  # type: ignore[import-untyped]
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion  # type: ignore[import-untyped]


class VideoTileDetailer:
    """Full-video refiner with optional reference image context.

    Processes the entire video latent in a single KSampler pass so the model's
    temporal attention operates across all frames simultaneously.  An optional
    reference_image is prepended as a frozen temporal frame (noise_mask=0) so
    it anchors style/identity across the sequence via temporal attention without
    itself being modified.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "How much to re-denoise. 0 = no change, 1 = full "
                            "re-generation. Keep low (0.1-0.3) to refine detail "
                            "without changing the original appearance."
                        ),
                    },
                ),
            },
            "optional": {
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Resized to frame dimensions and prepended as a frozen "
                            "temporal frame (noise_mask=0). Participates in temporal "
                            "attention across all video frames as a style/identity anchor."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Re-denoises a full video latent in one pass using pre-VACE model and "
        "conditioning, preserving the model's temporal attention across all frames. "
        "Optional reference_image is prepended as a frozen temporal frame that "
        "anchors style and identity through temporal attention."
    )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _fix_decoded(decoded: torch.Tensor, expected_h: int) -> torch.Tensor:
        """Normalise VAE output to (N, H, W, C)."""
        if decoded.ndim == 5:
            b, f, d1, d2, c = decoded.shape
            decoded = decoded.reshape(b * f, d1, d2, c)
        if decoded.shape[1] != expected_h:
            decoded = torch.rot90(decoded, k=3, dims=(1, 2)).contiguous()
        return decoded

    # ------------------------------------------------------------------ main

    def execute(
        self,
        latent,
        model,
        vae,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        reference_image=None,
    ):
        device = comfy.model_management.get_torch_device()

        lat = latent["samples"]  # (1, C, T, H_lat, W_lat)
        H_lat, W_lat = lat.shape[3], lat.shape[4]
        img_h, img_w = H_lat * 8, W_lat * 8
        vid_T = lat.shape[2]

        ref_T = 0
        if reference_image is not None:
            ref = (reference_image[0] if reference_image.ndim == 4 else reference_image).to(device)
            if ref.shape[0] != img_h or ref.shape[1] != img_w:
                ref = (
                    F.interpolate(
                        ref.unsqueeze(0).permute(0, 3, 1, 2),
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                )
            ref_lat = vae.encode(ref.unsqueeze(0)).to(device)  # (1, C, ref_T, H_lat, W_lat)
            ref_T = ref_lat.shape[2]

            # Prepend reference; noise_mask=0 keeps it frozen during denoising
            combined = torch.cat([ref_lat, lat], dim=2)
            ref_mask = torch.zeros(1, 1, ref_T, H_lat, W_lat, device=device)
            vid_mask = torch.ones(1, 1, vid_T, H_lat, W_lat, device=device)
            noise_mask = torch.cat([ref_mask, vid_mask], dim=2)
            latent_in = {"samples": combined, "noise_mask": noise_mask}

            if "denoise_mask_function" not in model.model_options:
                model = DifferentialDiffusion.execute(model)[0]

            print(f"[VideoTileDetailer] reference prepended: ref_T={ref_T}, vid_T={vid_T}")
        else:
            latent_in = {"samples": lat}

        start_step = int(steps * (1.0 - denoise))
        print(f"[VideoTileDetailer] sampling {vid_T} latent frames, start_step={start_step}/{steps}")

        sampled = nodes.NODE_CLASS_MAPPINGS["KSamplerAdvanced"]().sample(  # type: ignore[attr-defined]
            model,
            "enable",
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_in,
            start_step,
            steps,
            "disable",
        )[0]

        # Strip reference temporal frames then decode
        video_lat = sampled["samples"][:, :, ref_T:, :, :]
        decoded = vae.decode(video_lat)
        result = self._fix_decoded(decoded, img_h)
        print(f"[VideoTileDetailer] done → {result.shape}")

        return (result.clamp(0, 1).cpu(),)
