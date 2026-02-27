"""WanVaceToVideo variant with independent reference and control_video strengths.

Uses a model patch that scales the vace_strength per-token AFTER the VACE
attention blocks run. Reference frames (0:trim_latent) get full strength=1.0,
control frames (trim_latent:) get strength=control_video_strength. This operates
on the skip connection influence (c_skip) rather than distorting the input latent,
avoiding latent-space artifacts while maintaining clean pose separation.
"""

from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.utils


class WanVaceStrengthPatch:
    """Patches the model to apply per-frame strength scaling to VACE injection.

    Reference frames (0:trim_latent) are injected at full strength (1.0),
    control frames (trim_latent:) at reduced strength (control_video_strength).
    This scaling is applied to the skip connection AFTER the VACE attention
    blocks, so it controls the influence on the main denoising stream without
    distorting the latent space.

    Wire this into your model chain before KSampler. Connect trim_latent
    from WanVaceToVideo to the trim_latent input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "trim_latent": ("INT", {"default": 0, "min": 0, "max": 99999,
                                        "tooltip": "trim_latent output from WanVaceToVideo."}),
                "control_video_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Strength for control frame tokens (0=no pose "
                                "guidance from control video, 1=full)."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"
    DESCRIPTION = (
        "Per-frame VACE strength scaling. Reference frames at 1.0, control "
        "frames at specified strength. Wire trim_latent from WanVaceToVideo "
        "and insert before KSampler."
    )

    @classmethod
    def execute(cls, model, trim_latent, control_video_strength):
        if abs(control_video_strength - 1.0) < 1e-6:
            return (model.clone(),)

        m = model.clone()
        _ctrl = control_video_strength
        _trim = trim_latent

        def vace_unet_wrapper(apply_model, args):
            c = args["c"]
            vace_strength = c.get("vace_strength", None)

            # Only patch if vace_strength exists and is a list
            if vace_strength is None or not isinstance(vace_strength, list) or len(vace_strength) == 0:
                return apply_model(args["input"], args["timestep"], **c)

            # Extract latent shape to derive token geometry.
            # args["input"] shape: [batch, channels, T_latent, H_latent, W_latent]
            inp = args["input"]
            T = inp.shape[2]
            H = inp.shape[3]
            W = inp.shape[4]

            # vace_patch_embedding has kernel/stride = (1, 2, 2).
            # After embedding: [B, 2048, T_patches, H_patches, W_patches]
            # After flatten: [B, T_patches * H_patches * W_patches, 2048]
            h_p = H // 2
            w_p = W // 2
            tokens_per_frame = h_p * w_p
            total_tokens = T * tokens_per_frame

            # Build per-token strength mask: reference frames at 1.0, control at _ctrl.
            # Frame i spans token indices [i*tokens_per_frame : (i+1)*tokens_per_frame]
            strength_mask = torch.ones(1, total_tokens, 1, device=inp.device, dtype=inp.dtype)
            control_start_token = _trim * tokens_per_frame
            if control_start_token < total_tokens:
                strength_mask[0, control_start_token:, 0] = _ctrl

            # Replace scalar vace_strength with per-token tensor.
            # The model does: x += c_skip * vace_strength[iii]
            # Broadcasting: [B, seq_len, 2048] * [1, seq_len, 1] works.
            c_mod = dict(c)
            c_mod["vace_strength"] = [strength_mask]
            return apply_model(inp, args["timestep"], **c_mod)

        m.set_model_unet_function_wrapper(vace_unet_wrapper)
        return (m,)


class WanVaceToVideoControlStrength:
    """Stock WanVaceToVideo with trim_latent output for frame indexing.

    Pairs with WanVaceStrengthPatch. The strength parameter controls overall
    VACE guidance strength. WanVaceStrengthPatch then further reduces the
    control frame contribution via per-token scaling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 16384, "step": 16}),
                "height": (
                    "INT",
                    {"default": 480, "min": 16, "max": 16384, "step": 16},
                ),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
            },
            "optional": {
                "control_video": ("IMAGE",),
                "control_masks": ("MASK",),
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        strength,
        control_video=None,
        control_masks=None,
        reference_image=None,
    ):
        latent_length = ((length - 1) // 4) + 1

        if control_video is not None:
            control_video = comfy.utils.common_upscale(
                control_video[:length].movedim(-1, 1),
                width, height, "bilinear", "center",
            ).movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                )
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat(
                [reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))],
                dim=1,
            )

        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(
                mask[:length], width, height, "bilinear", "center"
            ).movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(
                    mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0
                )

        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact'
        ).squeeze(0)

        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)

        import node_helpers
        positive = node_helpers.conditioning_set_values(
            positive,
            {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]},
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]},
            append=True,
        )

        latent = torch.zeros(
            [batch_size, 16, latent_length, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return (positive, negative, {"samples": latent}, trim_latent)
