"""SEGS-based detailer for AnimateDiff with NAG compatibility and 5D tensor handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class DetailerByMask:
    """
    Complete replacement for DetailerForEachPipeForAnimateDiff with full NAG compatibility.

    This node combines:
    - SEGS crop region fixing (ensures divisibility by 64)
    - NAG-compatible upscaling (maintains divisibility during upscale)
    - 5D tensor normalization (prevents dimension errors)
    - Custom paste logic that handles video frames

    Use this instead of the standard DetailerForEachPipeForAnimateDiff when working
    with NAG-patched models to avoid autocast and dimension errors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_frames": ("IMAGE",),
                "segs": ("SEGS",),
                "guide_size": (
                    "FLOAT",
                    {"default": 512, "min": 64, "max": 8192, "step": 8},
                ),
                "guide_size_for": (
                    "BOOLEAN",
                    {"default": True, "label_on": "bbox", "label_off": "crop_region"},
                ),
                "max_size": (
                    "FLOAT",
                    {"default": 1024, "min": 64, "max": 8192, "step": 8},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (["euler", "euler_a", "dpmpp_2m", "dpmpp_sde"],),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform"],),
                "denoise": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01},
                ),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "basic_pipe": ("BASIC_PIPE",),
            },
            "optional": {
                "noise_mask_feather": (
                    "INT",
                    {"default": 20, "min": 0, "max": 100, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/Impact"
    DESCRIPTION = "SEGS-based detailer for AnimateDiff with NAG compatibility, automatic crop fixing, and 5D tensor handling"

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

        # Round UP to nearest multiple of divisor to avoid shrinking mask region
        width = ((width + divisor - 1) // divisor) * divisor
        height = ((height + divisor - 1) // divisor) * divisor

        # Recalculate x2, y2 from center to expand evenly
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

    def execute(
        self,
        image_frames,
        segs,
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
        noise_mask_feather=0,
    ):
        """Process SEGS with full NAG compatibility and 5D tensor handling."""
        try:
            from impact import utils
            from impact.animatediff_nodes import SEGSDetailerForAnimateDiff
            from impact.core import SEG
            from impact.utils import tensor_gaussian_blur_mask
        except ImportError:
            raise ImportError(
                "Could not import from Impact Pack. Make sure ComfyUI-Impact-Pack is installed."
            )

        # Hardcoded NAG divisor
        nag_divisor = 64

        # Get image dimensions
        img_height = image_frames.shape[1]
        img_width = image_frames.shape[2]

        # Process SEGS directly
        header, seg_list = segs
        fixed_seg_list = []

        for i, seg in enumerate(seg_list):
            crop_region = getattr(seg, "crop_region", None)

            if crop_region is not None:
                fixed_crop = self._fix_crop_region(
                    crop_region, img_width, img_height, nag_divisor
                )

                if fixed_crop != crop_region:
                    print(
                        f"[NAG Detailer] Seg {i}: Fixed crop_region {crop_region} → {fixed_crop} (divisor={nag_divisor})"
                    )

                # Replace seg with fixed crop
                if hasattr(seg, "_replace"):
                    fixed_seg = seg._replace(crop_region=fixed_crop)
                else:
                    # Fallback for non-namedtuple SEG
                    from impact.core import SEG

                    fixed_seg = SEG(
                        seg.cropped_image,
                        seg.cropped_mask,
                        seg.confidence,
                        fixed_crop,  # crop_region
                        seg.bbox,
                        seg.label,
                        seg.control_net_wrapper
                        if hasattr(seg, "control_net_wrapper")
                        else None,
                    )

                fixed_seg_list.append(fixed_seg)
            else:
                fixed_seg_list.append(seg)

        fixed_segs = (header, fixed_seg_list)

        # Create NAG-compatible detailer hook
        nag_hook = NAGDetailerHookImpl(nag_divisor)

        # Process each segment using Impact Pack's approach
        from impact.segs_nodes import SEGSPaste

        enhanced_segs = []
        cnet_image_list = []

        model, clip, vae, positive, negative = basic_pipe

        total_segments = len(fixed_seg_list)
        for seg_idx, sub_seg in enumerate(fixed_seg_list):
            print(
                f"[Detailer by Mask] Processing segment {seg_idx + 1}/{total_segments}"
            )
            seg = sub_seg

            # Use the segment's pre-cropped image if available, otherwise crop from source
            if seg.cropped_image is not None:
                # Segment already has cropped image (e.g., from Mask to SEGS)
                cropped_image_frames = utils.to_tensor(seg.cropped_image)
                if isinstance(cropped_image_frames, torch.Tensor):
                    cropped_image_frames = cropped_image_frames.cpu().numpy()
            else:
                # Need to crop from the full image_frames
                cropped_image_frames = None
                for image in image_frames:
                    image = image.unsqueeze(0)
                    cropped_image = utils.crop_tensor4(image, seg.crop_region)
                    cropped_image = utils.to_tensor(cropped_image)
                    if cropped_image_frames is None:
                        cropped_image_frames = cropped_image
                    else:
                        cropped_image_frames = torch.concat(
                            (cropped_image_frames, cropped_image), dim=0
                        )
                cropped_image_frames = cropped_image_frames.cpu().numpy()

            # Crop conditioning
            from impact import core
            from impact.core import SEG

            cropped_positive = [
                [
                    condition,
                    {
                        k: core.crop_condition_mask(
                            v, cropped_image_frames, seg.crop_region
                        )
                        if k == "mask"
                        else v
                        for k, v in details.items()
                    },
                ]
                for condition, details in positive
            ]

            cropped_negative = [
                [
                    condition,
                    {
                        k: core.crop_condition_mask(
                            v, cropped_image_frames, seg.crop_region
                        )
                        if k == "mask"
                        else v
                        for k, v in details.items()
                    },
                ]
                for condition, details in negative
            ]

            # Enhance detail with NAG hook
            # Detect if this is a single-frame segment or multi-frame batch
            is_single_frame = (
                isinstance(cropped_image_frames, np.ndarray)
                and cropped_image_frames.ndim == 4
                and cropped_image_frames.shape[0] == 1
            )

            if not (isinstance(model, str) and model == "DUMMY"):
                if is_single_frame:
                    # Single frame - use regular enhance_detail for better denoise behavior
                    cropped_image_tensor = torch.from_numpy(cropped_image_frames)
                    enhanced_image_tensor, cnet_images = core.enhance_detail(
                        cropped_image_tensor,
                        model,
                        clip,
                        vae,
                        guide_size,
                        guide_size_for,
                        max_size,
                        seg.bbox,
                        seed,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        cropped_positive,
                        cropped_negative,
                        denoise,
                        seg.cropped_mask,
                        force_inpaint=False,
                        refiner_ratio=None,
                        refiner_model=None,
                        refiner_clip=None,
                        refiner_positive=None,
                        refiner_negative=None,
                        control_net_wrapper=seg.control_net_wrapper
                        if hasattr(seg, "control_net_wrapper")
                        else None,
                        noise_mask_feather=noise_mask_feather,
                        scheduler_func=None,
                        detailer_hook=nag_hook,
                    )
                else:
                    # Multi-frame batch - use animatediff version
                    enhanced_image_tensor, cnet_images = (
                        core.enhance_detail_for_animatediff(
                            cropped_image_frames,
                            model,
                            clip,
                            vae,
                            guide_size,
                            guide_size_for,
                            max_size,
                            seg.bbox,
                            seed,
                            steps,
                            cfg,
                            sampler_name,
                            scheduler,
                            cropped_positive,
                            cropped_negative,
                            denoise,
                            seg.cropped_mask,
                            refiner_ratio=None,
                            refiner_model=None,
                            refiner_clip=None,
                            refiner_positive=None,
                            refiner_negative=None,
                            control_net_wrapper=seg.control_net_wrapper
                            if hasattr(seg, "control_net_wrapper")
                            else None,
                            noise_mask_feather=noise_mask_feather,
                            scheduler_func=None,
                            detailer_hook=nag_hook,
                        )
                    )
            else:
                enhanced_image_tensor = cropped_image_frames
                cnet_images = None

            if cnet_images is not None:
                cnet_image_list.extend(cnet_images)

            if enhanced_image_tensor is None:
                new_cropped_image = cropped_image_frames
            else:
                # Normalize to 4D if needed BEFORE converting to numpy
                if enhanced_image_tensor.ndim == 5:
                    # Flatten batch and frames dimensions: [B, F, H, W, C] -> [B*F, H, W, C]
                    b, f, h, w, c = enhanced_image_tensor.shape
                    enhanced_image_tensor = enhanced_image_tensor.reshape(
                        b * f, h, w, c
                    )
                    print(
                        f"[NAG Detailer] Normalized enhanced tensor from 5D {(b, f, h, w, c)} to 4D {enhanced_image_tensor.shape}"
                    )

                new_cropped_image = enhanced_image_tensor.cpu().numpy()

            new_seg = SEG(
                new_cropped_image,
                seg.cropped_mask,
                seg.confidence,
                seg.crop_region,
                seg.bbox,
                seg.label,
                None,
            )
            enhanced_seg = (header, [new_seg])

            # Paste the enhanced segment back onto the image
            image_frames = SEGSPaste.doit(
                image_frames, enhanced_seg, feather, alpha=255
            )[0]

            # Call NAG hook's post_paste
            image_frames = nag_hook.post_paste(image_frames)

            enhanced_segs.append(new_seg)

        # Collect all masks from enhanced segments
        output_mask = None
        for seg in enhanced_segs:
            if seg.cropped_mask is not None:
                if output_mask is None:
                    output_mask = seg.cropped_mask
                else:
                    # Combine masks if multiple segments
                    if isinstance(output_mask, torch.Tensor) and isinstance(
                        seg.cropped_mask, torch.Tensor
                    ):
                        output_mask = torch.max(output_mask, seg.cropped_mask)

        # If no mask found, create empty mask
        if output_mask is None:
            output_mask = torch.zeros((img_height, img_width), dtype=torch.float32)

        return (image_frames, output_mask)


class NAGDetailerHookImpl:
    """Internal NAG hook for dimension fixing during upscale."""

    def __init__(self, divisor: int = 64):
        self.divisor = divisor

    def touch_scaled_size(self, width: int, height: int) -> tuple[int, int]:
        """Round dimensions to nearest multiple of divisor."""
        adjusted_width = (width // self.divisor) * self.divisor
        adjusted_height = (height // self.divisor) * self.divisor

        if adjusted_width < self.divisor:
            adjusted_width = self.divisor
        if adjusted_height < self.divisor:
            adjusted_height = self.divisor

        if adjusted_width != width or adjusted_height != height:
            print(
                f"[NAG Hook] Adjusted upscale from ({width}, {height}) to ({adjusted_width}, {adjusted_height})"
            )

        return adjusted_width, adjusted_height

    def get_custom_sampler(self):
        """Return custom sampler if needed - None for default."""
        return None

    def post_encode(self, latent: dict) -> dict:
        """Called after VAE encoding - passthrough."""
        return latent

    def pre_decode(self, latent: dict) -> dict:
        """Called before VAE decoding - passthrough."""
        return latent

    def post_decode(self, image):
        """Called after VAE decoding - passthrough."""
        return image

    def post_paste(self, image):
        """Called after pasting enhanced segment - passthrough."""
        return image


__all__ = ["DetailerByMask"]
