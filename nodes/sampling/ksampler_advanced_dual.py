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
            },
            "optional": {
                "output_mode": (
                    ["both", "leftover_noise_only", "denoised_only"],
                    {
                        "default": "both",
                        "tooltip": (
                            "Both outputs runs the dual pass. The single-output "
                            "modes skip the unused sampling pass."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "latent_denoised")
    OUTPUT_TOOLTIPS = (
        "Selected latent output; in denoised_only mode this is fully denoised.",
        "Selected latent output; in leftover_noise_only mode this may include noise.",
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
        output_mode="both",
    ):
        if output_mode not in {"both", "leftover_noise_only", "denoised_only"}:
            output_mode = "both"

        disable_noise = add_noise == "disable"

        if output_mode == "denoised_only":
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
                denoise=1.0,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=True,
            )
            return (latent_denoised[0], latent_denoised[0])

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
            denoise=1.0,
            disable_noise=disable_noise,
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=False,
        )

        if output_mode == "leftover_noise_only":
            return (latent[0], latent[0])

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
                denoise=1.0,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=end_at_step,
                force_full_denoise=True,
            )
        else:
            latent_denoised = latent

        return (latent[0], latent_denoised[0])
