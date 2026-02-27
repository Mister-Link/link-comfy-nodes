"""WanVaceToVideo variant with independent reference and control_video strengths."""

from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.utils


def _make_vace_block_wrapper(original_forward, per_token_strength):
    """Wrap a VaceWanAttentionBlock.forward to scale c_skip per-token."""
    def wrapped(c, x, **kwargs):
        c_skip, c_out = original_forward(c, x, **kwargs)
        c_skip = c_skip * per_token_strength.to(device=c_skip.device, dtype=c_skip.dtype)
        return c_skip, c_out
    return wrapped


class WanVaceStrengthPatch:
    """Patches the model to apply independent transformer-space strengths to the
    reference_image and control_video regions of a WanVaceToVideo VACE stream.

    Wire this into your model chain before KSampler.  Connect trim_latent from
    WanVaceToVideo (or any VACE conditioning node) to the trim_latent input.

    At each denoising step the patch temporarily wraps the model's
    VaceWanAttentionBlock.forward methods so that c_skip is multiplied by a
    per-token strength tensor before being added to x.  Full-quality,
    on-distribution inputs are always passed to the VACE blocks — only the
    contribution weight changes.
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
                     "tooltip": "Strength of the reference_image VACE stream in transformer space."},
                ),
                "control_video_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Strength of the control_video VACE stream in transformer space. Reduce to loosen pose adherence."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"
    DESCRIPTION = (
        "Patches a WAN VACE model to apply separate transformer-space strengths "
        "to the reference_image and control_video regions. Wire trim_latent from "
        "WanVaceToVideo here and insert this node in your model chain before KSampler."
    )

    @classmethod
    def execute(cls, model, trim_latent, reference_strength, control_video_strength):
        needs_patch = trim_latent > 0 and (
            abs(reference_strength - 1.0) > 1e-6 or abs(control_video_strength - 1.0) > 1e-6
        )

        if not needs_patch:
            return (model.clone(),)

        m = model.clone()
        _base_model = m.model
        _ref_strength = reference_strength
        _ctrl_strength = control_video_strength
        _trim_latent = trim_latent

        def vace_unet_wrapper(apply_model, args):
            input_x = args["input"]
            # input_x: [batch, 16, T, H_latent, W_latent]
            T = input_x.shape[2]
            H_latent = input_x.shape[3]
            W_latent = input_x.shape[4]

            # vace_patch_embedding stride is (1, 2, 2):
            # T' = T, H' = H_latent // 2, W' = W_latent // 2
            tokens_per_frame = (H_latent // 2) * (W_latent // 2)
            total_tokens = T * tokens_per_frame
            ref_tokens = _trim_latent * tokens_per_frame

            per_token = torch.ones(1, total_tokens, 1)
            per_token[:, :ref_tokens, :] = _ref_strength
            per_token[:, ref_tokens:, :] = _ctrl_strength

            diff_model = _base_model.diffusion_model
            vace_blocks = getattr(diff_model, "vace_blocks", None)
            if vace_blocks is None:
                return apply_model(args["input"], args["timestep"], **args["c"])

            originals = {}
            try:
                for ii, block in enumerate(vace_blocks):
                    originals[ii] = block.forward
                    block.forward = _make_vace_block_wrapper(block.forward, per_token)
                result = apply_model(args["input"], args["timestep"], **args["c"])
            finally:
                for ii, orig in originals.items():
                    vace_blocks[ii].forward = orig

            return result

        m.set_model_unet_function_wrapper(vace_unet_wrapper)
        return (m,)


class WanVaceToVideoControlStrength:
    """WanVaceToVideo node that outputs trim_latent for use with WanVaceStrengthPatch.

    Identical to the stock WanVaceToVideo node.  Use WanVaceStrengthPatch in your
    model chain to apply independent transformer-space strengths to the reference
    and control regions.
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
