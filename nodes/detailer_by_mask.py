"""Mask-based detailer for AnimateDiff with NAG compatibility and 5D tensor handling."""

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
                "mask": ("MASK",),
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
    DESCRIPTION = "Mask-based detailer for AnimateDiff with NAG compatibility, automatic crop fixing, and 5D tensor handling"

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
        noise_mask_feather=0,
    ):
        """Process mask with full NAG compatibility and 5D tensor handling."""
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

        # Convert mask to SEGS
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Get mask bounding box
        mask_np = mask[0].cpu().numpy()
        rows = np.any(mask_np > 0.5, axis=1)
        cols = np.any(mask_np > 0.5, axis=0)

        if not rows.any() or not cols.any():
            # Empty mask, return original image with mask
            return (image_frames, mask)

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        bbox = (x1, y1, x2 + 1, y2 + 1)
        crop_region = bbox

        # Create a single SEG from the mask
        seg = SEG(
            cropped_image=None,
            cropped_mask=mask,
            confidence=1.0,
            crop_region=crop_region,
            bbox=bbox,
            label="mask",
            control_net_wrapper=None,
        )

        segs = (("", 0, 0), [seg])
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
            # Process segment with NAG hook by calling core functions directly
            from impact import core

            model, clip, vae, positive, negative = basic_pipe
            seg = sub_seg
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

            # Call enhance with NAG hook
            if not (isinstance(model, str) and model == "DUMMY"):
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

            if enhanced_image_tensor is None:
                new_cropped_image = cropped_image_frames
            else:
                new_cropped_image = enhanced_image_tensor.cpu().numpy()

            from impact.core import SEG

            enhanced_seg = (
                header,
                [
                    SEG(
                        new_cropped_image,
                        seg.cropped_mask,
                        seg.confidence,
                        seg.crop_region,
                        seg.bbox,
                        seg.label,
                        None,
                    )
                ],
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

        return (image_frames, mask)


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


__all__ = ["DetailerByMask"]
