from __future__ import annotations

import comfy.samplers
from nodes import common_ksampler


class KSamplerAdvancedDual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "latent_denoised")
    OUTPUT_TOOLTIPS = (
        "Latent that may include leftover noise (force_full_denoise=False).",
        "Fully denoised latent (force_full_denoise=True).",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "Advanced sampler that returns both the leftover-noise latent and the "
        "fully denoised latent."
    )

    def sample(
        self,
        model,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        denoise=1.0,
    ):
        disable_noise = add_noise == "disable"

        latent = common_ksampler(
            model,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=False,
        )

        # force_full_denoise only differs from the above when end_at_step
        # truncates the sigma schedule (samplers.py sets sigmas[-1] = 0).
        # When end_at_step >= steps the two outputs are identical, so skip
        # the second full sampling pass entirely.
        if end_at_step < steps:
            latent_denoised = common_ksampler(
                model,
                noise_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=True,
            )
        else:
            latent_denoised = latent

        return (latent[0], latent_denoised[0])
