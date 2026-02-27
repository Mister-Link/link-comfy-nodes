"""WanVaceToVideo variant with independent reference and control_video strengths.

Uses two separate VACE streams so that reference and control go through
vace_blocks independently (no cross-attention mixing).  Each stream gets
its own vace_strength, giving clean per-region control without skeleton
bleed from attention signal mixing.
"""

from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.utils


class WanVaceToVideoControlStrength:
    """WanVaceToVideo with separate reference and control strengths.

    When a reference_image is provided, two VACE streams are created:
      - Stream 0: reference frame + neutral padding, strength = reference_strength
      - Stream 1: neutral padding + control frames, strength = control_video_strength

    Each stream goes through vace_blocks independently so reference and
    control tokens never attend to each other.  This prevents skeleton
    bleed that occurs when per-token scaling is applied after attention
    has already mixed the signals.

    When no reference_image is provided, a single stream is used with
    control_video_strength as its vace_strength.
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
                "reference_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Strength of the reference_image VACE stream."},
                ),
                "control_video_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Strength of the control_video VACE stream. "
                                "Reduce to loosen pose adherence."},
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
        reference_strength,
        control_video_strength,
        control_video=None,
        control_masks=None,
        reference_image=None,
    ):
        latent_length = ((length - 1) // 4) + 1

        # --- process control_video (identical to stock) ---
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

        # --- process reference_image (identical to stock) ---
        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat(
                [reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))],
                dim=1,
            )

        # --- process control_masks (identical to stock) ---
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

        # --- encode inactive / reactive (identical to stock) ---
        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_latent = torch.cat((inactive, reactive), dim=1)

        # --- process mask into 64-channel VAE-stride form (identical to stock) ---
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact'
        ).squeeze(0)
        # mask: [64, latent_length, height_mask, width_mask]

        trim_latent = 0

        if reference_image is not None:
            ref_T = reference_image.shape[2]
            total_T = ref_T + latent_length
            trim_latent = ref_T

            H_latent = height // 8
            W_latent = width // 8

            # Neutral padding: process_out(zeros) = latent mean.
            # After process_latent_in in extra_conds this becomes zero
            # in model space, contributing no signal to attention.
            wan_fmt = comfy.latent_formats.Wan21()
            neutral_16 = wan_fmt.process_out(torch.zeros(1, 16, 1, H_latent, W_latent))
            neutral_32 = torch.cat([neutral_16, neutral_16], dim=1)

            # Stream 0 (reference): reference frame + neutral padding
            stream_ref = torch.cat([
                reference_image,
                neutral_32.expand(-1, -1, latent_length, -1, -1),
            ], dim=2)

            # Stream 1 (control): neutral padding + control frames
            stream_ctrl = torch.cat([
                neutral_32.expand(-1, -1, ref_T, -1, -1),
                control_latent,
            ], dim=2)

            # Reference mask: all zeros (reference convention)
            ref_mask = torch.zeros(1, 64, total_T, height_mask, width_mask)

            # Control mask: zeros for padding, control mask for control frames
            ctrl_mask = torch.cat([
                torch.zeros(64, ref_T, height_mask, width_mask),
                mask,
            ], dim=1).unsqueeze(0)

            vace_frames = [stream_ref, stream_ctrl]
            vace_masks = [ref_mask, ctrl_mask]
            vace_strengths = [reference_strength, control_video_strength]

            latent_length = total_T
        else:
            # No reference: single control stream
            vace_frames = [control_latent]
            vace_masks = [mask.unsqueeze(0)]
            vace_strengths = [control_video_strength]

        import node_helpers
        positive = node_helpers.conditioning_set_values(
            positive,
            {"vace_frames": vace_frames, "vace_mask": vace_masks, "vace_strength": vace_strengths},
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {"vace_frames": vace_frames, "vace_mask": vace_masks, "vace_strength": vace_strengths},
            append=True,
        )

        latent = torch.zeros(
            [batch_size, 16, latent_length, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return (positive, negative, {"samples": latent}, trim_latent)
