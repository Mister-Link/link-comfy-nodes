"""WanVaceToVideo variant with independent reference and control_video strengths."""

from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.utils


def _make_vace_block_wrapper(original_forward, per_token_strength):
    """Wrap a VaceWanAttentionBlock.forward to scale c_skip per-token.

    per_token_strength is a [1, total_tokens, 1] tensor with
    reference_strength for ref-region tokens and control_video_strength
    for control-region tokens.
    """

    def wrapped(c, x, **kwargs):
        c_skip, c_out = original_forward(c, x, **kwargs)
        c_skip = c_skip * per_token_strength.to(device=c_skip.device, dtype=c_skip.dtype)
        return c_skip, c_out

    return wrapped


class WanVaceToVideoControlStrength:
    """WanVaceToVideo with separate transformer-space strengths for reference and control.

    Patches the model's VaceWanAttentionBlock.forward methods so that
    c_skip is scaled per-token before being added to x.  The VACE blocks
    always see full-quality on-distribution inputs — only the contribution
    weight changes, preserving output quality at any strength.

    Requires MODEL as input so it can apply the patch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
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
                "reference_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Transformer-space strength for the reference_image region. Scales the VACE c_skip contribution for reference tokens.",
                    },
                ),
                "control_video_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Transformer-space strength for the control_video region. Reduce to loosen pose adherence while preserving quality.",
                    },
                ),
            },
            "optional": {
                "control_video": ("IMAGE",),
                "control_masks": ("MASK",),
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "trim_latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"
    DESCRIPTION = (
        "WanVaceToVideo with separate strength sliders for reference_image and "
        "control_video. Patches the model so VACE blocks always process "
        "full-quality inputs — only the c_skip contribution weight is scaled "
        "per-token in transformer space. This preserves output quality at any "
        "strength level."
    )

    @classmethod
    def execute(
        cls,
        model,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        reference_strength,
        control_video_strength,
        control_video=None,
        control_masks=None,
        reference_image=None,
    ):
        latent_length = ((length - 1) // 4) + 1

        # --- control_video preparation (same as stock WanVaceToVideo) ---
        if control_video is not None:
            control_video = comfy.utils.common_upscale(
                control_video[:length].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video,
                    (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]),
                    value=0.5,
                )
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        # --- reference_image (same as stock) ---
        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat(
                [
                    reference_image,
                    comfy.latent_formats.Wan21().process_out(
                        torch.zeros_like(reference_image)
                    ),
                ],
                dim=1,
            )

        # --- control_masks (same as stock) ---
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

        # --- inactive / reactive encode (same as stock) ---
        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if reference_image is not None:
            control_video_latent = torch.cat(
                (reference_image, control_video_latent), dim=2
            )

        # --- mask to latent space (same as stock) ---
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(
            vae_stride * vae_stride, length, height_mask, width_mask
        )
        mask = (
            torch.nn.functional.interpolate(
                mask.unsqueeze(0),
                size=(latent_length, height_mask, width_mask),
                mode="nearest-exact",
            )
            .squeeze(0)
        )

        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)

        # --- conditioning (single combined VACE stream, full strength) ---
        import node_helpers

        positive = node_helpers.conditioning_set_values(
            positive,
            {
                "vace_frames": [control_video_latent],
                "vace_mask": [mask],
                "vace_strength": [1.0],
            },
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {
                "vace_frames": [control_video_latent],
                "vace_mask": [mask],
                "vace_strength": [1.0],
            },
            append=True,
        )

        # --- model patch: per-token vace_strength ---
        m = model.clone()

        needs_patch = (
            reference_image is not None
            and (abs(reference_strength - 1.0) > 1e-6 or abs(control_video_strength - 1.0) > 1e-6)
        )

        if needs_patch:
            _ref_strength = reference_strength
            _ctrl_strength = control_video_strength
            _trim_latent = trim_latent
            _base_model = m.model  # BaseModel — holds diffusion_model

            def vace_unet_wrapper(apply_model, args):
                input_x = args["input"]
                # input_x shape: [batch, 16, T, H_latent, W_latent]
                T = input_x.shape[2]

                # After vace_patch_embedding (stride (1,2,2)):
                #   T' = T, H' = H_latent//2, W' = W_latent//2
                # Tokens are flattened in T × H' × W' order.
                H_latent = input_x.shape[3]
                W_latent = input_x.shape[4]
                tokens_per_frame = (H_latent // 2) * (W_latent // 2)
                total_tokens = T * tokens_per_frame
                ref_tokens = _trim_latent * tokens_per_frame

                per_token = torch.ones(1, total_tokens, 1)
                per_token[:, :ref_tokens, :] = _ref_strength
                per_token[:, ref_tokens:, :] = _ctrl_strength

                # Access diffusion model from captured BaseModel
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
        elif reference_image is None and abs(control_video_strength - 1.0) > 1e-6:
            # No reference — just use scalar vace_strength for control
            # Re-do conditioning with the actual strength
            positive = node_helpers.conditioning_set_values(
                positive,
                {"vace_strength": [control_video_strength]},
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {"vace_strength": [control_video_strength]},
            )

        latent = torch.zeros(
            [batch_size, 16, latent_length, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return (m, positive, negative, {"samples": latent}, trim_latent)
