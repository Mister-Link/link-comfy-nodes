"""Video detailer with dimension fixing and batch processing."""

from __future__ import annotations

import numpy as np
import torch

import comfy.sample
import comfy.samplers
import comfy.utils
import nodes


class VideoDetailer:
    """
    Detailer for video frames with NAG compatibility.

    This node:
    - Optionally accepts a MASK to detail specific regions
    - If no mask provided, details the entire image
    - Fixes crop regions to be divisible by 64 (required for NAG)
    - Processes all frames together as a batch
    - Does not depend on Impact Pack for sampling
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide_size": (
                    "FLOAT",
                    {"default": 720, "min": 64, "max": 8192, "step": 8},
                ),
                "guide_size_for": (
                    "BOOLEAN",
                    {"default": True, "label_on": "bbox", "label_off": "crop_region"},
                ),
                "max_size": (
                    "FLOAT",
                    {"default": 720, "min": 64, "max": 8192, "step": 8},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm"}),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"default": "simple"},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0001, "max": 1.0, "step": 0.01},
                ),
                "feather": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "basic_pipe": ("BASIC_PIPE",),
            },
            "optional": {
                "image_frames": ("IMAGE",),
                "latent": ("LATENT",),
                "mask_opt": ("MASK",),
                "noise_mask_feather": (
                    "INT",
                    {"default": 10, "min": 0, "max": 100, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Video detailer with NAG compatibility - processes all frames as a batch"
    )

    @staticmethod
    def _fix_to_divisor(value: int, divisor: int) -> int:
        """Round up to nearest multiple of divisor."""
        return ((value + divisor - 1) // divisor) * divisor

    @staticmethod
    def _fix_crop_region(
        crop_region: tuple, img_width: int, img_height: int, divisor: int
    ) -> tuple:
        """Fix crop region to be divisible by divisor and within image bounds."""
        x1, y1, x2, y2 = crop_region

        # Clamp to image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))

        width = x2 - x1
        height = y2 - y1

        # Round UP to nearest multiple of divisor
        width = ((width + divisor - 1) // divisor) * divisor
        height = ((height + divisor - 1) // divisor) * divisor

        # Recalculate from center to expand evenly
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        x1 = center_x - width // 2
        x2 = x1 + width
        y1 = center_y - height // 2
        y2 = y1 + height

        # Handle overflow by shifting
        if x2 > img_width:
            shift = x2 - img_width
            x1 -= shift
            x2 -= shift
        if x1 < 0:
            shift = -x1
            x1 += shift
            x2 += shift

        if y2 > img_height:
            shift = y2 - img_height
            y1 -= shift
            y2 -= shift
        if y1 < 0:
            shift = -y1
            y1 += shift
            y2 += shift

        # Final clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        return (x1, y1, x2, y2)

    @staticmethod
    def _get_mask_bbox(mask: torch.Tensor) -> tuple:
        """Get bounding box from mask. Returns (x1, y1, x2, y2)."""
        # Handle batched masks - combine them
        if mask.ndim == 3:
            combined = mask.max(dim=0)[0]
        else:
            combined = mask

        # Find non-zero pixels
        nonzero = torch.nonzero(combined > 0.5)
        if len(nonzero) == 0:
            h, w = combined.shape
            return (0, 0, w, h)

        y_coords = nonzero[:, 0]
        x_coords = nonzero[:, 1]

        y1 = int(y_coords.min().item())
        y2 = int(y_coords.max().item()) + 1
        x1 = int(x_coords.min().item())
        x2 = int(x_coords.max().item()) + 1

        return (x1, y1, x2, y2)

    @staticmethod
    def _gaussian_blur_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply gaussian blur to mask for feathering."""
        if kernel_size <= 0:
            return mask

        # Ensure odd kernel size
        kernel_size = kernel_size * 2 + 1

        # Check if mask is too small for kernel
        if mask.shape[-1] <= kernel_size or mask.shape[-2] <= kernel_size:
            kernel_size = min(mask.shape[-1], mask.shape[-2]) // 2
            if kernel_size % 2 == 0:
                kernel_size -= 1
            if kernel_size < 3:
                return mask

        # Create gaussian kernel
        sigma = kernel_size / 3.0
        x = (
            torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
            - kernel_size // 2
        )
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        # Apply blur
        padding = kernel_size // 2
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            blurred = torch.nn.functional.conv2d(mask, kernel_2d, padding=padding)
            return blurred.squeeze(0).squeeze(0)
        elif mask.ndim == 3:
            # [B, H, W] -> blur each
            result = []
            for m in mask:
                m = m.unsqueeze(0).unsqueeze(0)
                blurred = torch.nn.functional.conv2d(m, kernel_2d, padding=padding)
                result.append(blurred.squeeze(0).squeeze(0))
            return torch.stack(result)
        else:
            return mask

    def execute(
        self,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        feather,
        basic_pipe,
        image_frames=None,
        latent=None,
        mask_opt=None,
        noise_mask_feather=10,
    ):
        """Process all frames as a batch."""

        if image_frames is None and latent is None:
            raise ValueError(
                "[Video Detailer] Either image_frames or latent must be provided."
            )

        divisor = 64  # NAG divisor

        # Determine dimensions from whichever input is provided
        if image_frames is not None:
            num_frames = image_frames.shape[0]
            img_height = image_frames.shape[1]
            img_width = image_frames.shape[2]
        else:
            # Latent spatial dims are 1/8 of pixel dims
            latent_samples = latent["samples"]
            num_frames = latent_samples.shape[0]
            img_height = latent_samples.shape[2] * 8
            img_width = latent_samples.shape[3] * 8

        # If no mask provided, create a full mask
        if mask_opt is None:
            print(f"[Video Detailer] No mask provided, using full image")
            mask = torch.ones((num_frames, img_height, img_width), dtype=torch.float32)
        else:
            mask = mask_opt

        print(f"[Video Detailer] Processing {num_frames} frames as batch")

        # Handle mask dimensions
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(num_frames, -1, -1)
        elif mask.ndim == 3 and mask.shape[0] != num_frames:
            if mask.shape[0] == 1:
                mask = mask.expand(num_frames, -1, -1)
            else:
                mask = mask[0:1].expand(num_frames, -1, -1)

        print(f"[Video Detailer] Mask shape: {mask.shape}")

        # Get bounding box from combined mask
        bbox = self._get_mask_bbox(mask)
        print(f"[Video Detailer] Mask bbox: {bbox}")

        # Fix crop region for divisibility
        crop_region = self._fix_crop_region(bbox, img_width, img_height, divisor)
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        print(
            f"[Video Detailer] Fixed crop region: {crop_region} ({crop_width}x{crop_height})"
        )

        # Crop frames and masks (image path only — latent path never crops the latent)
        cropped_mask = mask[:, y1:y2, x1:x2]
        if image_frames is not None:
            cropped_frames = image_frames[:, y1:y2, x1:x2, :]
            print(f"[Video Detailer] Cropped frames shape: {cropped_frames.shape}")

        # Get model components from basic_pipe
        model, clip, vae, positive, negative = basic_pipe

        if image_frames is not None:
            # --- IMAGE PATH: crop → upscale → VAE encode → sample → decode → paste ---

            # Calculate upscale factor
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]

            if guide_size_for:  # bbox
                upscale = guide_size / min(bbox_w, bbox_h)
            else:  # crop_region
                upscale = guide_size / min(crop_width, crop_height)

            new_w = int(crop_width * upscale)
            new_h = int(crop_height * upscale)

            if new_w > max_size or new_h > max_size:
                upscale *= max_size / max(new_w, new_h)
                new_w = int(crop_width * upscale)
                new_h = int(crop_height * upscale)

            if upscale <= 1.0 or new_w == 0 or new_h == 0:
                upscale = 1.0
                new_w = crop_width
                new_h = crop_height

            new_w = self._fix_to_divisor(new_w, divisor)
            new_h = self._fix_to_divisor(new_h, divisor)
            print(
                f"[Video Detailer] Upscaling {crop_width}x{crop_height} -> {new_w}x{new_h}"
            )

            frames_nchw = cropped_frames.permute(0, 3, 1, 2)
            upscaled_frames = torch.nn.functional.interpolate(
                frames_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)

            upscaled_mask = torch.nn.functional.interpolate(
                cropped_mask.unsqueeze(1),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            if noise_mask_feather > 0:
                upscaled_mask = self._gaussian_blur_mask(
                    upscaled_mask, noise_mask_feather
                )

            print(f"[Video Detailer] VAE encoding {num_frames} frames...")
            latent_samples = vae.encode(upscaled_frames[:, :, :, :3])
            print(f"[Video Detailer] Latent shape: {latent_samples.shape}")

            latent_dict = {
                "samples": latent_samples,
                "noise_mask": upscaled_mask.unsqueeze(1),
            }
        else:
            # --- LATENT PATH: pass full latent unchanged; mask restricts denoising ---
            # VACE/WAN models require the latent shape to exactly match the conditioning
            # context, so we must never crop/resize the latent tensor itself.
            print(f"[Video Detailer] Latent input — passing full latent to sampler")
            latent_samples = latent["samples"]
            print(f"[Video Detailer] Latent shape: {latent_samples.shape}")

            # Build a full pixel-space noise mask (ComfyUI resizes it to latent space
            # internally). Shape: [F, 1, img_height, img_width].
            noise_mask = torch.zeros(
                (num_frames, 1, img_height, img_width),
                dtype=torch.float32,
                device=latent_samples.device,
            )

            # Upscale the cropped mask back to the crop region's pixel size and insert
            region_mask = torch.nn.functional.interpolate(
                cropped_mask.unsqueeze(1),
                size=(crop_height, crop_width),
                mode="bilinear",
                align_corners=False,
            )
            if noise_mask_feather > 0:
                region_mask = self._gaussian_blur_mask(
                    region_mask.squeeze(1), noise_mask_feather
                ).unsqueeze(1)
            noise_mask[:, :, y1:y2, x1:x2] = region_mask

            latent_dict = {
                "samples": latent_samples,
                "noise_mask": noise_mask,
            }

        # Sample using ComfyUI's native ksampler
        print(
            f"[Video Detailer] Sampling with {sampler_name}/{scheduler}, {steps} steps, denoise={denoise}..."
        )

        samples = nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_dict,
            denoise=denoise,
        )[0]

        # VAE decode
        print(f"[Video Detailer] VAE decoding...")
        decoded_frames = vae.decode(samples["samples"])
        print(f"[Video Detailer] Decoded shape: {decoded_frames.shape}")

        # Handle potential shape mismatches from VAE
        if decoded_frames.ndim == 5:
            b, f, h, w, c = decoded_frames.shape
            decoded_frames = decoded_frames.reshape(b * f, h, w, c)

        if image_frames is not None:
            # IMAGE PATH: decoded crop was upscaled — downscale back to original crop size,
            # then blend into the original frames.
            decoded_nchw = decoded_frames.permute(0, 3, 1, 2)
            downscaled_frames = torch.nn.functional.interpolate(
                decoded_nchw,
                size=(crop_height, crop_width),
                mode="bilinear",
                align_corners=False,
            )
            enhanced_crop_frames = downscaled_frames.permute(0, 2, 3, 1)
            print(f"[Video Detailer] Downscaled to: {enhanced_crop_frames.shape}")

            output_frames = image_frames.clone()
            for frame_idx in range(num_frames):
                frame_mask = cropped_mask[frame_idx].clone()
                if feather > 0:
                    frame_mask = self._gaussian_blur_mask(frame_mask, feather)

                enhanced_crop = (
                    enhanced_crop_frames[frame_idx]
                    if frame_idx < enhanced_crop_frames.shape[0]
                    else enhanced_crop_frames[-1]
                )

                frame_mask_expanded = frame_mask.unsqueeze(-1).to(output_frames.device)
                enhanced_crop = enhanced_crop.to(output_frames.device)
                original_crop = output_frames[frame_idx, y1:y2, x1:x2, :]

                blended = (
                    original_crop * (1 - frame_mask_expanded)
                    + enhanced_crop * frame_mask_expanded
                )
                output_frames[frame_idx, y1:y2, x1:x2, :] = blended
        else:
            # LATENT PATH: decoded_frames is already full-resolution (same shape as
            # decoding the original latent). Blend the detailed region back using the
            # original decoded latent as the base.
            print(f"[Video Detailer] Decoding original latent for blending base...")
            original_decoded = vae.decode(latent["samples"])
            if original_decoded.ndim == 5:
                b, f, h, w, c = original_decoded.shape
                original_decoded = original_decoded.reshape(b * f, h, w, c)

            output_frames = original_decoded.clone()
            for frame_idx in range(num_frames):
                frame_mask = cropped_mask[frame_idx].clone()
                if feather > 0:
                    frame_mask = self._gaussian_blur_mask(frame_mask, feather)

                enhanced_crop = (
                    decoded_frames[frame_idx]
                    if frame_idx < decoded_frames.shape[0]
                    else decoded_frames[-1]
                )
                enhanced_crop = enhanced_crop[y1:y2, x1:x2, :]

                frame_mask_expanded = frame_mask.unsqueeze(-1).to(output_frames.device)
                enhanced_crop = enhanced_crop.to(output_frames.device)
                original_crop = output_frames[frame_idx, y1:y2, x1:x2, :]

                blended = (
                    original_crop * (1 - frame_mask_expanded)
                    + enhanced_crop * frame_mask_expanded
                )
                output_frames[frame_idx, y1:y2, x1:x2, :] = blended

        print(f"[Video Detailer] Done - processed {num_frames} frames")

        # Return output
        output_mask = mask[0] if mask.ndim == 3 else mask
        return (output_frames, output_mask)


__all__ = ["VideoDetailer"]
