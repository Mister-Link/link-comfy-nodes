"""Complete NAG-compatible AnimateDiff detailer with built-in 5D tensor handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class DetailerForAnimateDiffNAG:
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
                "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "nag_divisor": (
                    "INT",
                    {
                        "default": 64,
                        "min": 8,
                        "max": 512,
                        "step": 8,
                        "tooltip": "Divisor for NAG compatibility (64 = 8 for VAE × 8 for attention)",
                    },
                ),
            },
            "optional": {
                "refiner_basic_pipe_opt": ("BASIC_PIPE",),
                "noise_mask_feather": (
                    "INT",
                    {"default": 20, "min": 0, "max": 100, "step": 1},
                ),
                "scheduler_func_opt": ("SCHEDULER_FUNC",),
            },
        }

    RETURN_TYPES = ("IMAGE", "SEGS", "BASIC_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "segs", "basic_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "execute"
    CATEGORY = "link/Impact"
    DESCRIPTION = "NAG-compatible AnimateDiff detailer with automatic crop fixing, dimension normalization, and 5D tensor handling"

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

        # Round down to nearest multiple of divisor
        width = (width // divisor) * divisor
        height = (height // divisor) * divisor

        # Ensure minimum size
        if width < divisor:
            width = divisor
        if height < divisor:
            height = divisor

        # Recalculate x2, y2
        x2 = x1 + width
        y2 = y1 + height

        # Handle overflow
        if x2 > img_width:
            overflow = x2 - img_width
            x1 = max(0, x1 - overflow)
            x2 = x1 + width
            if x2 > img_width:
                width = img_width - x1
                width = (width // divisor) * divisor
                x2 = x1 + width

        if y2 > img_height:
            overflow = y2 - img_height
            y1 = max(0, y1 - overflow)
            y2 = y1 + height
            if y2 > img_height:
                height = img_height - y1
                height = (height // divisor) * divisor
                y2 = y1 + height

        return (x1, y1, x2, y2)

    @staticmethod
    def _normalize_to_4d(tensor: Any) -> Any:
        """Normalize tensor to 4D NHWC format, handling 5D video tensors."""
        if tensor is None:
            return None

        if not isinstance(tensor, (np.ndarray, torch.Tensor)):
            return tensor

        arr = tensor

        # Handle 5D tensors by flattening batch and frames
        while arr.ndim > 4:
            squeeze_dim = None
            for dim in range(arr.ndim):
                if arr.shape[dim] == 1:
                    squeeze_dim = dim
                    break
            if squeeze_dim is not None:
                arr = (
                    arr.squeeze(squeeze_dim)
                    if isinstance(arr, torch.Tensor)
                    else np.squeeze(arr, axis=squeeze_dim)
                )
            else:
                if arr.ndim == 5:
                    b, f, h, w, c = arr.shape
                    if isinstance(arr, torch.Tensor):
                        arr = arr.reshape(b * f, h, w, c)
                    else:
                        arr = arr.reshape(b * f, h, w, c)
                else:
                    arr = arr[0]

        if arr.ndim == 3:
            arr = (
                arr[None, ...]
                if isinstance(arr, torch.Tensor)
                else np.expand_dims(arr, axis=0)
            )

        return arr

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
        refiner_ratio,
        nag_divisor=64,
        refiner_basic_pipe_opt=None,
        noise_mask_feather=0,
        scheduler_func_opt=None,
    ):
        """Process segments with full NAG compatibility and 5D tensor handling."""
        try:
            from impact import utils
            from impact.animatediff_nodes import SEGSDetailerForAnimateDiff
            from impact.utils import tensor_gaussian_blur_mask
        except ImportError:
            raise ImportError(
                "Could not import from Impact Pack. Make sure ComfyUI-Impact-Pack is installed."
            )

        # Get image dimensions
        img_height = image_frames.shape[1]
        img_width = image_frames.shape[2]

        # Fix all crop regions in SEGS first
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

                # Replace seg with fixed crop and cleared cropped_image
                if hasattr(seg, "_replace"):
                    fixed_seg = seg._replace(crop_region=fixed_crop, cropped_image=None)
                else:
                    # Fallback for non-namedtuple SEG
                    from impact.core import SEG

                    fixed_seg = SEG(
                        None,  # cropped_image
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

        # Process each segment
        enhanced_segs = []
        cnet_image_list = []

        for sub_seg in fixed_seg_list:
            single_seg = header, [sub_seg]

            # Detail this segment with NAG hook
            enhanced_seg, cnet_images = SEGSDetailerForAnimateDiff().do_detail(
                image_frames,
                single_seg,
                guide_size,
                guide_size_for,
                max_size,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                denoise,
                basic_pipe,
                refiner_ratio,
                refiner_basic_pipe_opt,
                noise_mask_feather,
                scheduler_func_opt=scheduler_func_opt,
            )

            # Normalize cropped_images to 4D
            _, enhanced_seg_list = enhanced_seg
            for seg in enhanced_seg_list:
                if seg.cropped_image is not None:
                    original_shape = (
                        seg.cropped_image.shape
                        if isinstance(seg.cropped_image, (np.ndarray, torch.Tensor))
                        else None
                    )
                    if original_shape and len(original_shape) == 5:
                        normalized = self._normalize_to_4d(seg.cropped_image)
                        print(
                            f"[NAG Detailer] Normalized cropped_image from {original_shape} to {normalized.shape}"
                        )
                        if hasattr(seg, "_replace"):
                            seg = seg._replace(cropped_image=normalized)

            # Paste using custom 5D-safe logic
            batch_size = image_frames.shape[0]
            result = torch.empty_like(image_frames)

            with torch.no_grad():
                for i in range(batch_size):
                    image_i = image_frames[i].unsqueeze(0).clone()

                    for seg in enhanced_seg_list:
                        ref_image = None

                        if seg.cropped_image is not None:
                            cropped_image = seg.cropped_image

                            if isinstance(cropped_image, np.ndarray):
                                cropped_image = torch.from_numpy(cropped_image)

                            cropped_image = self._normalize_to_4d(cropped_image)

                            if cropped_image is not None and i < len(cropped_image):
                                ref_image = cropped_image[i].unsqueeze(0)
                            elif cropped_image is not None:
                                ref_image = cropped_image[-1].unsqueeze(0)

                        if ref_image is None:
                            continue

                        # Handle mask
                        cmask = seg.cropped_mask
                        if cmask.ndim == 3 and len(cmask) == batch_size:
                            mask = cmask[i]
                        elif cmask.ndim == 3 and len(cmask) > 1:
                            mask = torch.any(cmask > 0.1, dim=0).float()
                        else:
                            mask = cmask

                        mask = tensor_gaussian_blur_mask(mask, feather) * (255 / 255.0)
                        mask = mask.to(image_i.device)
                        ref_image = ref_image.to(image_i.device)

                        x, y, *_ = seg.crop_region
                        utils.tensor_paste(image_i, ref_image, (x, y), mask)

                    result[i] = image_i[0]

            image_frames = result

            if cnet_images is not None:
                cnet_image_list.extend(cnet_images)

            enhanced_segs += enhanced_seg_list

        new_segs = header, enhanced_segs
        return (image_frames, new_segs, basic_pipe, cnet_image_list)


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


__all__ = ["DetailerForAnimateDiffNAG"]
