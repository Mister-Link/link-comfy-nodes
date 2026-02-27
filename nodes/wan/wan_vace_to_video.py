"""WanVaceToVideo variant with independent reference and control_video strengths."""

from __future__ import annotations

import torch

import comfy.latent_formats
import comfy.model_management
import comfy.utils


class WanVaceToVideoControlStrength:
    """WanVaceToVideo with separate strength inputs for reference_image and control_video.

    The model's VaceWanModel.forward_orig() iterates over each VACE stream
    independently and applies its own vace_strength scalar:

        for iii in range(len(c)):
            c_skip, c[iii] = self.vace_blocks[ii](c[iii], ...)
            x += c_skip * vace_strength[iii]

    This node splits reference and control into two separate VACE streams
    (two entries in the vace_frames list), each with its own vace_strength,
    so their strengths are fully decoupled in transformer space.

    control_video_neutral:
        Background pixel value of the control_video input (0-1).
        At control_video_strength < 1.0 the background is pushed toward 0.5
        (latent silence) rather than producing artifacts.
        Use 0.5 for gray-background VACE inputs (default).
        Use 0.0 for black-background OpenPose frames.
        No effect at control_video_strength=1.0.
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
                "reference_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Transformer-space strength for the reference_image VACE stream."},
                ),
                "control_video_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Transformer-space strength for the control_video VACE stream. Reduce to loosen pose adherence."},
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
        "control_video. Each is injected as an independent VACE stream so their "
        "strengths are fully decoupled in transformer space. Reduce "
        "control_video_strength to loosen pose/motion adherence without "
        "affecting the reference frame."
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
                control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                )
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        # Pixel-space attenuation of control_video toward 0.5 (latent silence).
        # At strength=1.0 this is a no-op.
        if abs(control_video_strength - 1.0) > 1e-6:
            control_video = 0.5 + (control_video - control_video_neutral) * control_video_strength

        # --- reference_image ---
        ref_latent = None
        trim_latent = 0
        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            ref_latent = vae.encode(reference_image[:, :, :, :3])
            ref_latent = torch.cat(
                [ref_latent, comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_latent))],
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
        ctrl_mask = ctrl_mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        ctrl_mask = torch.nn.functional.interpolate(
            ctrl_mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode="nearest-exact"
        ).squeeze(0).unsqueeze(0)

        # --- reference latent mask (zeros = "already known, don't inpaint") ---
        ref_mask = None
        if ref_latent is not None:
            ref_mask = torch.zeros(1, 64, ref_latent.shape[2], height_mask, width_mask)

        # --- Build two separate VACE streams ---
        # Each stream gets its own vace_strength, applied independently in
        # VaceWanModel.forward_orig()'s loop: x += c_skip * vace_strength[iii]
        import node_helpers

        if ref_latent is not None:
            # Stream 1: reference_image
            positive = node_helpers.conditioning_set_values(
                positive,
                {"vace_frames": [ref_latent], "vace_mask": [ref_mask], "vace_strength": [reference_strength]},
                append=True,
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {"vace_frames": [ref_latent], "vace_mask": [ref_mask], "vace_strength": [reference_strength]},
                append=True,
            )
            latent_length_out = latent_length + trim_latent
        else:
            latent_length_out = latent_length

        # Stream 2: control_video
        positive = node_helpers.conditioning_set_values(
            positive,
            {"vace_frames": [ctrl_latent], "vace_mask": [ctrl_mask], "vace_strength": [control_video_strength]},
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {"vace_frames": [ctrl_latent], "vace_mask": [ctrl_mask], "vace_strength": [control_video_strength]},
            append=True,
        )

        latent = torch.zeros(
            [batch_size, 16, latent_length_out, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return (positive, negative, {"samples": latent}, trim_latent)
