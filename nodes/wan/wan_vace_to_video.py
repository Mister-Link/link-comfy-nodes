"""WanVaceToVideo variant with independent reference and control_video strengths."""

from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.utils


class WanVaceToVideoControlStrength:
    """WanVaceToVideo with separate strength inputs for reference_image and control_video.

    Both strengths work via pixel-space attenuation before VAE encoding:
    each input is lerped toward 0.5 (latent silence) independently, then
    combined into a single VACE stream.

    Separate transformer-space streams are not possible because
    torch.stack in model_base.py requires all VACE streams to have the
    same temporal dimension (reference=1 frame vs control=latent_length).

    control_video_neutral:
        The background color of the control_video in 0-1 pixel space.
        Used to lerp the control_video toward silence when control_video_strength < 1.
        After the lerp the pipeline subtracts 0.5, so the effective silence point is
        always 0.5 in pixel space.  Set neutral=0.5 for gray-bg VACE inputs (default).
        Set neutral=0.0 for black-background OpenPose frames — at strength<1 the black
        background is pushed toward 0.5 (silence) while the skeleton is attenuated.
        At strength=1.0 this is a no-op regardless of neutral.
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
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Attenuates reference_image toward silence (0.5) before VAE encoding. At 0.0 the reference is fully silenced.",
                    },
                ),
                "control_video_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Attenuates control_video toward silence (0.5) before VAE encoding. Reduce to loosen pose adherence.",
                    },
                ),
                "control_video_neutral": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Background color of control_video in 0-1 pixel space. "
                            "Use 0.5 for gray-background VACE inputs. "
                            "Use 0.0 for black-background OpenPose frames — the black "
                            "background is lerped toward 0.5 (latent silence) as "
                            "control_video_strength decreases. No effect at strength=1.0."
                        ),
                    },
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
        "WanVaceToVideo with separate strength sliders for reference_image and "
        "control_video. Each is attenuated independently in pixel space before "
        "VAE encoding — reducing a strength fades that input toward silence "
        "without affecting the other."
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
        reference_strength,
        control_video_strength,
        control_video_neutral=0.5,
        control_video=None,
        control_masks=None,
        reference_image=None,
    ):
        latent_length = ((length - 1) // 4) + 1

        # --- control_video preparation ---
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

        # Attenuate control_video toward 0.5 (latent silence after - 0.5 centering).
        # At strength=1.0 this is always a no-op.
        if abs(control_video_strength - 1.0) > 1e-6:
            offset = 0.5 - control_video_neutral * control_video_strength
            control_video = offset + control_video * control_video_strength

        # --- reference_image ---
        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            # Pixel-space attenuation of reference toward 0.5 (latent silence).
            if abs(reference_strength - 1.0) > 1e-6:
                reference_image = 0.5 + (reference_image - 0.5) * reference_strength
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
        cv = control_video - 0.5
        inactive = (cv * (1 - mask)) + 0.5
        reactive = (cv * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)

        # Prepend reference_image along temporal dim (single combined VACE stream)
        if reference_image is not None:
            control_video_latent = torch.cat(
                (reference_image, control_video_latent), dim=2
            )

        # --- mask to latent space ---
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

        # --- conditioning (single combined VACE stream) ---
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

        latent = torch.zeros(
            [batch_size, 16, latent_length, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return (positive, negative, {"samples": latent}, trim_latent)
