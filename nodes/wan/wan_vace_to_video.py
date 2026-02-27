"""WanVaceToVideo variant with independent control_video strength."""

from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.utils


class WanVaceToVideoControlStrength:
    """WanVaceToVideo with a separate control_video_strength input.

    Identical to the core WanVaceToVideo node except that control_video frames
    are lerped toward the neutral value (0.5) by control_video_strength before
    the inactive/reactive VAE encode. This is the correct place to attenuate
    control_video influence:

      effective = neutral + (control_video - neutral) * control_video_strength
                = 0.5 + (control_video - 0.5) * control_video_strength

    At 1.0 the result is identical to WanVaceToVideo. At 0.0 the control_video
    is fully suppressed (pure neutral gray — no pose/motion signal). The
    reference_image and control_masks paths are unaffected.

    The core WanVaceToVideo `strength` input is preserved and still controls
    the global vace_strength scalar applied in the model forward pass.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 16384, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 16384, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "control_video_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "control_video_neutral": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Value to fade toward at strength=0. Use 0.5 for general VACE inputs, 0.0 for black-background OpenPose frames."},
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
        "WanVaceToVideo with a separate control_video_strength slider. "
        "Attenuates the control_video signal before VAE encoding by lerping "
        "toward control_video_neutral. Use neutral=0.5 for general VACE inputs; "
        "use neutral=0.0 for black-background OpenPose frames. "
        "Reference image and masks are unaffected."
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
        control_video_strength,
        control_video_neutral=0.5,
        control_video=None,
        control_masks=None,
        reference_image=None,
    ):
        # --- control_video preparation (mirrors WanVaceToVideo exactly) ---
        latent_length = ((length - 1) // 4) + 1

        if control_video is not None:
            control_video = comfy.utils.common_upscale(
                control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                )
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        # --- Attenuate control_video toward neutral before encoding ---
        # At strength=1.0 this is a no-op.
        # neutral=0.5 for general VACE; neutral=0.0 for black-bg OpenPose.
        if abs(control_video_strength - 1.0) > 1e-6:
            neutral = torch.full_like(control_video, control_video_neutral)
            control_video = neutral + (control_video - neutral) * control_video_strength

        # --- reference_image ---
        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat(
                [reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))],
                dim=1,
            )

        # --- control_masks ---
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

        # --- inactive / reactive encode ---
        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)

        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        # --- mask to latent space ---
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
            mask_pad = torch.zeros_like(mask[:, : reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)

        # --- conditioning ---
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
