"""WanVaceToVideo variant with independent reference and control_video strengths."""

from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.utils


class WanVaceToVideoControlStrength:
    """WanVaceToVideo with separate transformer-space strengths for reference and control.

    When a reference_image is provided, builds two VACE streams padded to
    equal temporal length so torch.stack succeeds:

        ref_stream:  [ref_latent | zeros_padding]   strength = reference_strength
        ctrl_stream: [zeros_padding | ctrl_latent]   strength = control_video_strength

    Each stream's c_skip is scaled independently in VaceWanModel.forward_orig:
        x += c_skip * vace_strength[iii]

    The zero-padded regions produce near-zero c_skip contributions because
    both latent and mask are zero ("known nothing"), minimizing cross-talk.

    control_video_neutral:
        Background pixel value of the control_video input (0-1).
        Remaps background toward 0.5 before VAE encoding to avoid artifacts.
        Use 0.5 for gray-background VACE inputs (default).
        Use 0.0 for black-background OpenPose frames.
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
                        "tooltip": "Transformer-space strength for the reference_image VACE stream.",
                    },
                ),
                "control_video_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Transformer-space strength for the control_video VACE stream. Reduce to loosen pose adherence.",
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
                            "background is remapped toward 0.5 before VAE encoding."
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
        "control_video. When both are present, each is a separate VACE stream "
        "with independent transformer-space strength. Reduce "
        "control_video_strength to loosen pose adherence without affecting "
        "the reference frame."
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

        # Remap control_video background toward 0.5 to avoid VAE artifacts.
        # At neutral=0.5 this is identity. At neutral=0.0 (black bg OpenPose),
        # remaps black -> 0.5 so VAE sees silence for the background.
        if abs(control_video_neutral - 0.5) > 1e-6:
            control_video = 0.5 + (control_video - control_video_neutral)

        # --- reference_image ---
        ref_latent = None
        trim_latent = 0
        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            ref_latent = vae.encode(reference_image[:, :, :, :3])
            ref_latent = torch.cat(
                [
                    ref_latent,
                    comfy.latent_formats.Wan21().process_out(
                        torch.zeros_like(ref_latent)
                    ),
                ],
                dim=1,
            )
            trim_latent = ref_latent.shape[2]

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

        # --- mask to latent space ---
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        ctrl_mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        ctrl_mask = ctrl_mask.permute(2, 4, 0, 1, 3)
        ctrl_mask = ctrl_mask.reshape(
            vae_stride * vae_stride, length, height_mask, width_mask
        )
        ctrl_mask = (
            torch.nn.functional.interpolate(
                ctrl_mask.unsqueeze(0),
                size=(latent_length, height_mask, width_mask),
                mode="nearest-exact",
            )
            .squeeze(0)
        )
        # ctrl_mask shape: [64, latent_length, height_mask, width_mask]

        import node_helpers

        if ref_latent is not None:
            # --- Two padded VACE streams of equal temporal length ---
            total_t = latent_length + trim_latent

            # Reference stream: [ref_latent | zero padding]
            # pad temporal dim (dim 2 of 5D) on the right by latent_length
            ref_stream = torch.nn.functional.pad(
                ref_latent, (0, 0, 0, 0, 0, latent_length)
            )
            # Ref mask: all zeros (= "known") — ref region is known content,
            # padding region is known silence (zeros). Shape: [1, 64, total_t, H, W]
            ref_mask = torch.zeros(1, 64, total_t, height_mask, width_mask)

            # Control stream: [zero padding | ctrl_latent]
            # pad temporal dim on the left by trim_latent
            ctrl_stream = torch.nn.functional.pad(
                ctrl_latent, (0, 0, 0, 0, trim_latent, 0)
            )
            # Control mask: zeros for padding region, ctrl_mask for control region
            ctrl_mask_full = torch.nn.functional.pad(
                ctrl_mask, (0, 0, 0, 0, trim_latent, 0)
            )
            ctrl_mask_full = ctrl_mask_full.unsqueeze(0)

            positive = node_helpers.conditioning_set_values(
                positive,
                {
                    "vace_frames": [ref_stream, ctrl_stream],
                    "vace_mask": [ref_mask, ctrl_mask_full],
                    "vace_strength": [reference_strength, control_video_strength],
                },
                append=True,
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {
                    "vace_frames": [ref_stream, ctrl_stream],
                    "vace_mask": [ref_mask, ctrl_mask_full],
                    "vace_strength": [reference_strength, control_video_strength],
                },
                append=True,
            )

            latent_length_out = total_t
        else:
            # --- Single VACE stream (control only) ---
            ctrl_mask = ctrl_mask.unsqueeze(0)

            positive = node_helpers.conditioning_set_values(
                positive,
                {
                    "vace_frames": [ctrl_latent],
                    "vace_mask": [ctrl_mask],
                    "vace_strength": [control_video_strength],
                },
                append=True,
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {
                    "vace_frames": [ctrl_latent],
                    "vace_mask": [ctrl_mask],
                    "vace_strength": [control_video_strength],
                },
                append=True,
            )

            latent_length_out = latent_length

        latent = torch.zeros(
            [batch_size, 16, latent_length_out, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return (positive, negative, {"samples": latent}, trim_latent)
