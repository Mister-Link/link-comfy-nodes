"""Mask-based detailer for video with dimension fixing and batch processing."""

from __future__ import annotations

import impact.core as core
import numpy as np
import torch

import comfy.samplers


class DetailerByMask:
    """
    Detailer for video frames using a mask input with NAG compatibility.

    This node:
    - Accepts a MASK instead of SEGS for simpler workflows
    - Fixes crop regions to be divisible by 64 (required for NAG)
    - Processes all frames together as a batch
    - Handles 5D tensor normalization to prevent dimension errors
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_frames": ("IMAGE",),
                "mask": ("MASK",),
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
                "scheduler": (core.get_schedulers(), {"default": "simple"}),
                "denoise": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0001, "max": 1.0, "step": 0.01},
                ),
                "feather": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "basic_pipe": ("BASIC_PIPE",),
            },
            "optional": {
                "noise_mask_feather": (
                    "INT",
                    {"default": 10, "min": 0, "max": 100, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/Impact"
    DESCRIPTION = "Mask-based detailer for video with NAG compatibility - processes all frames as a batch"

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
            # [B, H, W] - take max across batch to get union of all masks
            combined = mask.max(dim=0)[0]
        else:
            combined = mask

        # Find non-zero pixels
        nonzero = torch.nonzero(combined > 0.5)
        if len(nonzero) == 0:
            # No mask content, return full image
            h, w = combined.shape
            return (0, 0, w, h)

        y_coords = nonzero[:, 0]
        x_coords = nonzero[:, 1]

        y1 = int(y_coords.min().item())
        y2 = int(y_coords.max().item()) + 1
        x1 = int(x_coords.min().item())
        x2 = int(x_coords.max().item()) + 1

        return (x1, y1, x2, y2)

    def execute(
        self,
        image_frames,
        mask,
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
        noise_mask_feather=10,
    ):
        """Process all frames as a batch using the mask."""
        try:
            from impact import utils
            from impact.utils import tensor_gaussian_blur_mask
        except ImportError:
            raise ImportError(
                "Could not import from Impact Pack. Make sure ComfyUI-Impact-Pack is installed."
            )

        nag_divisor = 64
        num_frames = image_frames.shape[0]
        img_height = image_frames.shape[1]
        img_width = image_frames.shape[2]

        print(f"[Detailer by Mask] Processing {num_frames} frames as batch")

        # Handle mask dimensions
        # mask can be [H, W], [B, H, W], or [F, H, W] where F = num_frames
        if mask.ndim == 2:
            # Single mask for all frames
            mask = mask.unsqueeze(0).expand(num_frames, -1, -1)
        elif mask.ndim == 3 and mask.shape[0] != num_frames:
            # Mask batch doesn't match frame count - use first or expand
            if mask.shape[0] == 1:
                mask = mask.expand(num_frames, -1, -1)
            else:
                # Take first mask and expand
                mask = mask[0:1].expand(num_frames, -1, -1)

        print(f"[Detailer by Mask] Mask shape: {mask.shape}")

        # Get bounding box from combined mask
        bbox = self._get_mask_bbox(mask)
        print(f"[Detailer by Mask] Mask bbox: {bbox}")

        # Fix crop region for NAG divisibility
        crop_region = self._fix_crop_region(bbox, img_width, img_height, nag_divisor)
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        print(
            f"[Detailer by Mask] Fixed crop region: {crop_region} ({crop_width}x{crop_height})"
        )

        # Crop all frames
        cropped_frames = image_frames[:, y1:y2, x1:x2, :]
        print(f"[Detailer by Mask] Cropped frames shape: {cropped_frames.shape}")

        # Crop mask to same region
        cropped_mask = mask[:, y1:y2, x1:x2]
        print(f"[Detailer by Mask] Cropped mask shape: {cropped_mask.shape}")

        # Convert to numpy for enhance_detail_for_animatediff
        cropped_frames_np = cropped_frames.cpu().numpy()

        # Get single mask for the detailer (use first frame's mask or combined)
        single_mask = cropped_mask[0].cpu().numpy()

        model, clip, vae, positive, negative = basic_pipe

        # Create dimension hook
        dimension_hook = DivisibleDimensionHook(nag_divisor)

        # Calculate bbox relative to crop region
        relative_bbox = (0, 0, crop_width, crop_height)

        # Process all frames together
        print(f"[Detailer by Mask] Running enhance_detail_for_animatediff...")
        enhanced_tensor, cnet_images = core.enhance_detail_for_animatediff(
            cropped_frames_np,
            model,
            clip,
            vae,
            guide_size,
            guide_size_for,
            max_size,
            relative_bbox,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            denoise,
            single_mask,
            refiner_ratio=None,
            refiner_model=None,
            refiner_clip=None,
            refiner_positive=None,
            refiner_negative=None,
            control_net_wrapper=None,
            noise_mask_feather=noise_mask_feather,
            scheduler_func=None,
            detailer_hook=dimension_hook,
        )

        if enhanced_tensor is None:
            print("[Detailer by Mask] No enhancement performed, returning original")
            return (image_frames, mask[0])

        # Handle 5D tensor output
        if enhanced_tensor.ndim == 5:
            b, f, h, w, c = enhanced_tensor.shape
            enhanced_tensor = enhanced_tensor.reshape(b * f, h, w, c)
            print(f"[Detailer by Mask] Reshaped from 5D to {enhanced_tensor.shape}")

        print(f"[Detailer by Mask] Enhanced tensor shape: {enhanced_tensor.shape}")

        # Check if enhanced tensor needs to be resized back to crop dimensions
        enhanced_h = enhanced_tensor.shape[1]
        enhanced_w = enhanced_tensor.shape[2]

        if enhanced_h != crop_height or enhanced_w != crop_width:
            print(
                f"[Detailer by Mask] Resizing enhanced tensor from {enhanced_h}x{enhanced_w} to {crop_height}x{crop_width}"
            )
            # Resize: [F, H, W, C] -> [F, C, H, W] for interpolate, then back
            enhanced_tensor = enhanced_tensor.permute(0, 3, 1, 2)
            enhanced_tensor = torch.nn.functional.interpolate(
                enhanced_tensor,
                size=(crop_height, crop_width),
                mode="bilinear",
                align_corners=False,
            )
            enhanced_tensor = enhanced_tensor.permute(0, 2, 3, 1)
            print(f"[Detailer by Mask] Resized to {enhanced_tensor.shape}")

        # Paste enhanced regions back
        output_frames = image_frames.clone()

        for frame_idx in range(num_frames):
            # Get feathered mask for this frame
            frame_mask = cropped_mask[frame_idx]
            frame_mask = tensor_gaussian_blur_mask(frame_mask, feather)

            # Get enhanced crop for this frame
            if frame_idx < enhanced_tensor.shape[0]:
                enhanced_crop = enhanced_tensor[frame_idx]
            else:
                # Fallback if frame count mismatch
                enhanced_crop = enhanced_tensor[-1]

            # Blend
            frame_mask_expanded = frame_mask.unsqueeze(-1).to(output_frames.device)
            enhanced_crop = enhanced_crop.to(output_frames.device)
            original_crop = output_frames[frame_idx, y1:y2, x1:x2, :]

            blended = (
                original_crop * (1 - frame_mask_expanded)
                + enhanced_crop * frame_mask_expanded
            )
            output_frames[frame_idx, y1:y2, x1:x2, :] = blended

        print(f"[Detailer by Mask] Done - processed {num_frames} frames")

        # Return the original mask (first frame or combined)
        output_mask = mask[0] if mask.ndim == 3 else mask

        return (output_frames, output_mask)


class DivisibleDimensionHook:
    """Ensures dimensions are divisible by a given value (e.g., 64 for NAG models)."""

    def __init__(self, divisor: int = 64):
        self.divisor = divisor

    def touch_scaled_size(self, width: int, height: int) -> tuple[int, int]:
        adjusted_width = (width // self.divisor) * self.divisor
        adjusted_height = (height // self.divisor) * self.divisor

        if adjusted_width < self.divisor:
            adjusted_width = self.divisor
        if adjusted_height < self.divisor:
            adjusted_height = self.divisor

        if adjusted_width != width or adjusted_height != height:
            print(
                f"[DivisibleDimensionHook] Adjusted from ({width}, {height}) to ({adjusted_width}, {adjusted_height})"
            )

        return adjusted_width, adjusted_height

    def get_custom_sampler(self):
        return None

    def post_encode(self, latent):
        return latent

    def pre_decode(self, latent):
        return latent

    def post_decode(self, image):
        return image

    def post_paste(self, image):
        return image

    def post_upscale(self, image, mask):
        return image

    def get_skip_sampling(self):
        return False

    def set_steps(self, info):
        pass

    def cycle_latent(self, latent):
        return latent

    def pre_ksample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise,
    ):
        return (
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise,
        )

    def get_custom_noise(self, seed, noise, is_touched=False):
        return None, False

    def post_detection(self, segs):
        return segs

    def post_crop_region(self, w, h, bbox, crop_region):
        return crop_region


__all__ = ["DetailerByMask"]
