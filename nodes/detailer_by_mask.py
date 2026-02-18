"""Video detailer — reference-guided refinement using VACE conditioning."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

import comfy.latent_formats
import comfy.model_management
import comfy.samplers
import comfy.utils
import node_helpers
import nodes


class VideoDetailer:
    """
    Reference-guided video detailer using VACE conditioning.

    Stitches a reference image beside each video frame, then uses the VACE
    model to denoise the frame half while the reference half stays frozen.
    The model "sees" the reference and keeps the output consistent with it.

    Pipeline:
      1. Decode video latent → pixel frames
      2. Build composite video: [reference | frame] per frame (double-width)
      3. Build VACE conditioning: left=inactive (frozen ref), right=reactive
      4. KSampler denoises the double-width latent
      5. Decode, crop out right halves (the refined frames)
      6. Feathered alpha-blend paste onto original frames
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "basic_pipe": ("BASIC_PIPE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0001, "max": 1.0, "step": 0.01},
                ),
                "feather": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Gaussian blur radius for the paste-back mask edge.",
                    },
                ),
                "noise_mask_feather": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Gaussian blur radius for the latent noise mask "
                        "(DifferentialDiffusion boundary softness).",
                    },
                ),
                "vace_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "VACE conditioning strength for reference guidance.",
                    },
                ),
            },
            "optional": {
                "mask_opt": ("MASK",),
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image to guide denoising. The model "
                        "sees this as frozen context and keeps output "
                        "consistent with it."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Reference-guided video detailer. Stitches reference image beside "
        "each frame, uses VACE to denoise with the reference as frozen "
        "context. Unmasked areas are never modified."
    )

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _gaussian_blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Gaussian blur a 2D or 3D (batch of 2D) mask tensor."""
        if radius <= 0:
            return mask

        kernel_size = radius * 2 + 1
        min_dim = min(mask.shape[-1], mask.shape[-2])
        if min_dim <= kernel_size:
            kernel_size = min_dim // 2
            if kernel_size % 2 == 0:
                kernel_size -= 1
            if kernel_size < 3:
                return mask

        sigma = kernel_size / 3.0
        x = (
            torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
            - kernel_size // 2
        )
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        k1d = gauss / gauss.sum()
        k2d = (k1d.unsqueeze(0) * k1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        pad = kernel_size // 2

        def _blur_one(m2d):
            return (
                F.conv2d(m2d.unsqueeze(0).unsqueeze(0), k2d, padding=pad)
                .squeeze(0)
                .squeeze(0)
            )

        if mask.ndim == 2:
            return _blur_one(mask)
        return torch.stack([_blur_one(m) for m in mask])

    @staticmethod
    def _fix_decoded_shape(decoded: torch.Tensor, expected_height: int) -> torch.Tensor:
        """Normalise WAN VAE output [B,F,d1,d2,C] → [F,H,W,C]."""
        if decoded.ndim == 5:
            b, f, d1, d2, c = decoded.shape
            decoded = decoded.reshape(b * f, d1, d2, c)
        if decoded.shape[1] != expected_height:
            decoded = torch.rot90(decoded, k=3, dims=(1, 2)).contiguous()
        return decoded

    @staticmethod
    def _strip_vace_keys(conditioning):
        """Return a copy of conditioning with VACE keys removed."""
        out = []
        for tensor, meta in conditioning:
            meta = meta.copy()
            meta.pop("vace_frames", None)
            meta.pop("vace_mask", None)
            meta.pop("vace_strength", None)
            out.append([tensor, meta])
        return out

    def _build_vace_conditioning(
        self,
        vae,
        composite_video,
        vace_mask_pixel,
        num_pixel_frames,
        height,
        width_double,
        vace_strength,
        positive,
        negative,
        reference_image_latent=None,
    ):
        """Build VACE conditioning from a composite video and mask.

        Args:
            vae: the WAN VAE
            composite_video: [F, H, W*2, C] pixel frames (ref|frame side by side)
            vace_mask_pixel: [F, H, W*2, 1] mask (0=frozen, 1=reactive)
            num_pixel_frames: number of pixel frames
            height, width_double: spatial dims of composite
            vace_strength: VACE strength float
            positive, negative: conditioning to attach VACE to
            reference_image_latent: optional [1, 32, 1, H//8, W_double//8] prepend ref
        Returns:
            (positive, negative, latent_length)
        """
        latent_length = ((num_pixel_frames - 1) // 4) + 1

        # Inactive/reactive split (VACE's core trick)
        # Grey (0.5) = "no information" signal
        cv = composite_video - 0.5
        inactive = (cv * (1 - vace_mask_pixel)) + 0.5  # frozen regions show content
        reactive = (cv * vace_mask_pixel) + 0.5  # reactive regions show content

        # VAE encode both halves
        inactive_latent = vae.encode(inactive[:, :, :, :3])
        reactive_latent = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive_latent, reactive_latent), dim=1)
        # Shape: [1, 32, latent_length, H//8, W_double//8]

        # Optionally prepend reference image on temporal axis
        if reference_image_latent is not None:
            control_video_latent = torch.cat(
                (reference_image_latent, control_video_latent), dim=2
            )

        # Build the 64-channel sub-pixel mask
        vae_stride = 8
        h_mask = height // vae_stride
        w_mask = width_double // vae_stride

        # vace_mask_pixel: [F, H, W_double, 1] → [F, H, W_double]
        mask_flat = vace_mask_pixel.squeeze(-1)
        # Reshape to unfold 8×8 sub-pixels
        mask_r = mask_flat.view(
            num_pixel_frames, h_mask, vae_stride, w_mask, vae_stride
        )
        mask_r = mask_r.permute(2, 4, 0, 1, 3)  # [8, 8, F, H//8, W//8]
        mask_r = mask_r.reshape(
            vae_stride * vae_stride, num_pixel_frames, h_mask, w_mask
        )  # [64, F, H//8, W//8]

        mask_r = F.interpolate(
            mask_r.unsqueeze(0),
            size=(latent_length, h_mask, w_mask),
            mode="nearest-exact",
        ).squeeze(0)  # [64, LT, H//8, W//8]

        # Prepend zeros for reference frame if present
        if reference_image_latent is not None:
            ref_temporal = reference_image_latent.shape[2]
            mask_pad = torch.zeros(64, ref_temporal, h_mask, w_mask)
            mask_r = torch.cat((mask_pad, mask_r), dim=1)
            latent_length += ref_temporal

        vace_mask = mask_r.unsqueeze(0)  # [1, 64, total_LT, H//8, W//8]

        # Strip old VACE keys and attach new ones
        new_positive = self._strip_vace_keys(positive)
        new_negative = self._strip_vace_keys(negative)

        vace_values = {
            "vace_frames": [control_video_latent],
            "vace_mask": [vace_mask],
            "vace_strength": [vace_strength],
        }
        new_positive = node_helpers.conditioning_set_values(
            new_positive, vace_values, append=True
        )
        new_negative = node_helpers.conditioning_set_values(
            new_negative, vace_values, append=True
        )

        return new_positive, new_negative, latent_length

    # ---------------------------------------------- main entry point

    def execute(
        self,
        latent,
        basic_pipe,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        feather,
        noise_mask_feather,
        vace_strength,
        mask_opt=None,
        reference_image=None,
    ):
        model, clip, vae, positive, negative = basic_pipe

        # WAN latent: [B, C, F, H, W]
        latent_samples = latent["samples"]
        img_height = latent_samples.shape[3] * 8
        img_width = latent_samples.shape[4] * 8

        print(
            f"[Video Detailer] latent {latent_samples.shape} → {img_width}×{img_height} px"
        )

        # --- Step 1: Decode the original video latent → pixel frames ---
        original_decoded = vae.decode(latent_samples)
        original_frames = self._fix_decoded_shape(original_decoded, img_height)
        num_frames = original_frames.shape[0]
        print(f"[Video Detailer] decoded {num_frames} frames {original_frames.shape}")

        # --- Step 2: Build pixel-space mask [F, H, W] ---
        if mask_opt is None:
            mask = torch.ones((num_frames, img_height, img_width), dtype=torch.float32)
        else:
            mask = mask_opt.clone()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(num_frames, -1, -1).contiguous()
            elif mask.shape[0] != num_frames:
                mask = mask[0:1].expand(num_frames, -1, -1).contiguous()

        # --- Step 3: Prepare reference image ---
        if reference_image is not None:
            ref = reference_image
            # Take first frame only
            if ref.ndim == 4:
                ref = ref[0]  # [H, W, C]
            # Resize to match video frame dimensions
            if ref.shape[0] != img_height or ref.shape[1] != img_width:
                ref = ref.unsqueeze(0).permute(0, 3, 1, 2)  # [1,C,H,W]
                ref = F.interpolate(
                    ref,
                    size=(img_height, img_width),
                    mode="bilinear",
                    align_corners=False,
                )
                ref = ref.permute(0, 2, 3, 1).squeeze(0)  # [H,W,C]
            print(f"[Video Detailer] reference image {ref.shape}")
        else:
            # Use first frame as reference
            ref = original_frames[0]
            print("[Video Detailer] using first decoded frame as reference")

        # --- Step 4: Build composite video [ref | frame] per frame ---
        # Double-width: left half = reference (frozen), right half = frame (reactive)
        width_double = img_width * 2
        # Round to multiple of 16 for VAE compatibility
        width_double = ((width_double + 15) // 16) * 16

        composite_frames = []
        for i in range(num_frames):
            frame = original_frames[i]  # [H, W, C]
            # Pad reference and frame to half of width_double each
            half_w = width_double // 2
            ref_resized = (
                F.interpolate(
                    ref.unsqueeze(0).permute(0, 3, 1, 2),
                    size=(img_height, half_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .squeeze(0)
            )
            frame_resized = (
                F.interpolate(
                    frame.unsqueeze(0).permute(0, 3, 1, 2),
                    size=(img_height, half_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .squeeze(0)
            )

            composite = torch.cat([ref_resized, frame_resized], dim=1)  # [H, W*2, C]
            composite_frames.append(composite)

        composite_video = torch.stack(composite_frames)  # [F, H, W*2, C]
        print(f"[Video Detailer] composite video {composite_video.shape}")

        # --- Step 5: Build VACE mask ---
        # Left half (reference) = 0 (inactive/frozen)
        # Right half (frame) = 1 (reactive/denoisable)
        half_w = width_double // 2
        vace_mask_pixel = torch.zeros(
            (num_frames, img_height, width_double, 1), dtype=torch.float32
        )
        # Apply the user's mask to the right half only
        for i in range(num_frames):
            frame_mask = mask[i]  # [H, W]
            # Resize mask to match half_w if needed
            if frame_mask.shape[1] != half_w:
                fm = (
                    F.interpolate(
                        frame_mask.unsqueeze(0).unsqueeze(0),
                        size=(img_height, half_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            else:
                fm = frame_mask
            vace_mask_pixel[i, :, half_w:, 0] = fm

        print(
            f"[Video Detailer] VACE mask {vace_mask_pixel.shape} "
            f"mean={vace_mask_pixel.mean():.3f}"
        )

        # --- Step 6: Encode reference image for VACE temporal prepend ---
        ref_for_encode = ref.unsqueeze(0)  # [1, H, W, C]
        # Resize to double-width to match composite
        ref_wide = F.interpolate(
            ref_for_encode.permute(0, 3, 1, 2),
            size=(img_height, width_double),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)  # [1, H, W_double, C]

        ref_encoded = vae.encode(ref_wide[:, :, :, :3])
        # [1, 16, 1, H//8, W_double//8]
        ref_null = comfy.latent_formats.Wan21().process_out(
            torch.zeros_like(ref_encoded)
        )
        reference_image_latent = torch.cat([ref_encoded, ref_null], dim=1)
        # [1, 32, 1, H//8, W_double//8]

        # --- Step 7: Build VACE conditioning ---
        new_positive, new_negative, total_latent_length = self._build_vace_conditioning(
            vae,
            composite_video,
            vace_mask_pixel,
            num_frames,
            img_height,
            width_double,
            vace_strength,
            positive,
            negative,
            reference_image_latent=reference_image_latent,
        )
        print(
            f"[Video Detailer] VACE conditioning built, "
            f"latent_length={total_latent_length}"
        )

        # --- Step 8: Build the denoising latent ---
        # VACE is designed to work from a zero/noise latent — it provides
        # the guidance, the sampler generates from scratch following it.
        # Use denoise=1.0 for full VACE-guided generation, or lower to
        # blend between the encoded content and VACE-guided generation.
        latent_h = img_height // 8
        latent_w = width_double // 8
        denoise_latent = torch.zeros(
            [1, 16, total_latent_length, latent_h, latent_w],
            device=comfy.model_management.intermediate_device(),
        )
        print(
            f"[Video Detailer] denoise latent {denoise_latent.shape} (zeros — VACE guides generation)"
        )

        # No noise_mask / DifferentialDiffusion needed here — the VACE mask
        # already tells the model which regions are frozen vs reactive.
        # Adding a noise_mask would fight with the VACE conditioning.
        latent_dict = {"samples": denoise_latent}

        # Sanity check
        assert denoise_latent.shape[2] == total_latent_length, (
            f"Latent temporal dim {denoise_latent.shape[2]} != "
            f"expected {total_latent_length}"
        )

        # --- Step 9: KSampler ---
        samples = nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            new_positive,
            new_negative,
            latent_dict,
            denoise=denoise,
        )[0]

        # --- Step 10: Decode and extract right halves ---
        # Trim the reference temporal frame first
        trimmed = samples["samples"][:, :, 1:, :, :]  # remove ref frame
        decoded = vae.decode(trimmed)
        decoded_frames = self._fix_decoded_shape(decoded, img_height)
        print(f"[Video Detailer] decoded refined {decoded_frames.shape}")

        # Extract right half of each frame
        refined_frames = decoded_frames[:, :, half_w:, :]
        # Resize back to original width if needed
        if refined_frames.shape[2] != img_width:
            rf = refined_frames.permute(0, 3, 1, 2)  # [F,C,H,W]
            rf = F.interpolate(
                rf,
                size=(img_height, img_width),
                mode="bilinear",
                align_corners=False,
            )
            refined_frames = rf.permute(0, 2, 3, 1)  # [F,H,W,C]

        print(f"[Video Detailer] refined frames {refined_frames.shape}")

        # --- Step 11: Pixel-space compositing ---
        composite_mask = self._gaussian_blur_mask(mask, feather)
        mask_4d = composite_mask.unsqueeze(-1).to(original_frames.device)

        # Match frame counts (decoded may differ due to temporal upscaling)
        out_frames = min(original_frames.shape[0], refined_frames.shape[0])
        output = original_frames[:out_frames].clone()
        output = (1 - mask_4d[:out_frames]) * output + mask_4d[
            :out_frames
        ] * refined_frames[:out_frames].to(output.device)

        print(f"[Video Detailer] output {output.shape}")
        output_mask = mask[0] if mask.ndim == 3 else mask
        return (output, output_mask)


__all__ = ["VideoDetailer"]
