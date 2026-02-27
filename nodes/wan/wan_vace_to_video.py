"""WanVaceToVideo variant with independent reference and control_video strengths."""

from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.utils


class WanVaceToVideoControlStrength:
    """WanVaceToVideo with separate strengths for reference_image and control_video.

    Uses latent-space attenuation: each input is VAE-encoded normally, then
    lerped toward its silence latent (VAE-encoded 0.5 pixels) before being
    combined into a single VACE stream.  This avoids the torch.stack shape
    mismatch of separate streams and gives a true strength gradient (unlike
    pixel-space attenuation which was binary due to VAE nonlinearity).

        attenuated = silence_latent + (actual_latent - silence_latent) * strength

    At strength=1.0 this is identity (no extra VAE work).
    At strength=0.0 the input is fully replaced with silence.
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
                        "tooltip": "Latent-space strength for the reference_image. Lerps toward silence at <1.0.",
                    },
                ),
                "control_video_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Latent-space strength for the control_video. Lerps toward silence at <1.0. Reduce to loosen pose adherence.",
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
        "control_video. Each is attenuated independently in latent space by "
        "lerping toward its VAE-encoded silence (0.5 pixels). This gives a "
        "true strength gradient — reduce control_video_strength to loosen "
        "pose adherence without affecting the reference frame."
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

        # --- reference_image ---
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

            # Latent-space attenuation: lerp toward silence (VAE-encoded 0.5 pixels)
            if abs(reference_strength - 1.0) > 1e-6:
                silence_ref_pixels = torch.ones((1, height, width, 3)) * 0.5
                silence_ref = vae.encode(silence_ref_pixels)
                silence_ref = torch.cat(
                    [
                        silence_ref,
                        comfy.latent_formats.Wan21().process_out(
                            torch.zeros_like(silence_ref)
                        ),
                    ],
                    dim=1,
                )
                reference_image = silence_ref + (reference_image - silence_ref) * reference_strength

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
        ctrl_latent = torch.cat((inactive, reactive), dim=1)

        # Latent-space attenuation: lerp toward silence (VAE-encoded 0.5 pixels)
        if abs(control_video_strength - 1.0) > 1e-6:
            silence_pixels = torch.ones((length, height, width, 3)) * 0.5
            silence_enc = vae.encode(silence_pixels)
            silence_ctrl = torch.cat((silence_enc, silence_enc), dim=1)
            ctrl_latent = silence_ctrl + (ctrl_latent - silence_ctrl) * control_video_strength

        # --- combine into single stream (same as stock WanVaceToVideo) ---
        if reference_image is not None:
            ctrl_latent = torch.cat((reference_image, ctrl_latent), dim=2)

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

        # --- conditioning (single combined VACE stream, full strength) ---
        import node_helpers

        positive = node_helpers.conditioning_set_values(
            positive,
            {
                "vace_frames": [ctrl_latent],
                "vace_mask": [mask],
                "vace_strength": [1.0],
            },
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {
                "vace_frames": [ctrl_latent],
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
