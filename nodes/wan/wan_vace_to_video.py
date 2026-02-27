"""WanVaceToVideo variant with independent reference and control_video strengths."""

from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.utils


class WanVaceStrengthPatch:
    """Patches the model to apply independent strengths to the reference_image
    and control_video regions of a WanVaceToVideo VACE stream.

    Wire this into your model chain before KSampler.  Connect trim_latent from
    WanVaceToVideo (or any VACE conditioning node) to the trim_latent input.

    At each denoising step the patch modifies the vace_context tensor before
    it reaches vace_patch_embedding:

      - reference frames (temporal indices 0..trim_latent-1):
          image channels (0:16) are blended toward reference baseline channels
          (16:32) with reference_strength, avoiding noisy attenuation from
          scaling latents toward zero.

      - control frames (temporal indices trim_latent..end):
          REACTIVE channels (16:32, pose/mask signal) are blended toward
          INACTIVE channels (0:16, background baseline) using
          control_video_strength. This avoids hard thresholding that can
          happen when scaling reactive channels toward zero.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "trim_latent": ("INT", {"default": 0, "min": 0, "max": 99999,
                                        "tooltip": "trim_latent output from WanVaceToVideo."}),
                "reference_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Strength of the reference_image VACE stream."},
                ),
                "control_video_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Strength of the pose/mask signal in the control_video VACE stream. "
                                "Reduce to loosen pose adherence while keeping the background intact."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"
    DESCRIPTION = (
        "Patches a WAN VACE model to apply separate strengths to the "
        "reference_image and control_video regions. Both reference and control "
        "strengths blend signal channels toward baseline channels for smooth "
        "attenuation without noisy thresholding. "
        "Wire trim_latent from WanVaceToVideo here and insert this node in "
        "your model chain before KSampler."
    )

    @classmethod
    def execute(cls, model, trim_latent, reference_strength, control_video_strength):
        needs_ref_patch = trim_latent > 0 and abs(reference_strength - 1.0) > 1e-6
        needs_ctrl_patch = abs(control_video_strength - 1.0) > 1e-6

        if not needs_ref_patch and not needs_ctrl_patch:
            return (model.clone(),)

        m = model.clone()
        _ref_strength = reference_strength
        _ctrl_strength = control_video_strength
        _trim_latent = trim_latent

        def vace_unet_wrapper(apply_model, args):
            c = args["c"]
            vace_ctx = c.get("vace_context", None)

            if vace_ctx is None:
                return apply_model(args["input"], args["timestep"], **c)

            # vace_ctx: [batch_chunks, n_streams, C, T, H_latent, W_latent]
            # C = 96: channels 0:16 = inactive (background),
            #          channels 16:32 = reactive (pose/mask signal),
            #          channels 32:96 = mask
            # Temporal layout (dim 3): [0:trim_latent] = reference frames,
            #                          [trim_latent:] = control frames

            c_modified = dict(c)
            vace_ctx = vace_ctx.clone()

            if needs_ref_patch and _trim_latent > 0:
                # Blend reference image channels toward reference baseline
                # channels instead of scaling all channels toward zero.
                ref_image_ctx = vace_ctx[:, 0, 0:16, :_trim_latent, :, :]
                ref_base_ctx = vace_ctx[:, 0, 16:32, :_trim_latent, :, :]
                vace_ctx[:, 0, 0:16, :_trim_latent, :, :] = (
                    ref_base_ctx + ((ref_image_ctx - ref_base_ctx) * _ref_strength)
                )

            if needs_ctrl_patch:
                # Blend reactive pose channels toward inactive/background channels
                # instead of scaling toward zero. At 0.0 this becomes inactive;
                # at 1.0 it is unchanged.
                inactive_ctx = vace_ctx[:, 0, 0:16, _trim_latent:, :, :]
                reactive_ctx = vace_ctx[:, 0, 16:32, _trim_latent:, :, :]
                vace_ctx[:, 0, 16:32, _trim_latent:, :, :] = (
                    inactive_ctx + ((reactive_ctx - inactive_ctx) * _ctrl_strength)
                )

            c_modified["vace_context"] = vace_ctx
            return apply_model(args["input"], args["timestep"], **c_modified)

        m.set_model_unet_function_wrapper(vace_unet_wrapper)
        return (m,)


class WanVaceToVideoControlStrength:
    """WanVaceToVideo node that outputs trim_latent for use with WanVaceStrengthPatch.

    Identical to the stock WanVaceToVideo node.  Use WanVaceStrengthPatch in your
    model chain to apply independent reference_strength and control_video_strength.
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
                    {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                     "tooltip": "Overall vace_strength for the combined VACE stream."},
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
    DESCRIPTION = (
        "WanVaceToVideo. Use WanVaceStrengthPatch in your model chain to apply "
        "independent reference_strength and control_video_strength."
    )

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
            mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode="nearest-exact"
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
