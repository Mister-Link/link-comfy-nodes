"""SEGS fixer nodes to handle dimension issues with Impact Pack nodes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _normalize_tensor_to_4d(tensor: Any) -> Any:
    """Normalize a tensor to 4D NHWC format.

    For 5D tensors in format [batch, frames, height, width, channels],
    flattens batch and frames into a single batch dimension.
    """
    if tensor is None:
        return None

    if not isinstance(tensor, (np.ndarray, torch.Tensor)):
        return tensor

    arr = tensor

    # Squeeze singleton dimensions until we reach 4D
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
            # If 5D with no singleton dims, assume [batch, frames, H, W, C]
            # Flatten batch and frames: [batch*frames, H, W, C]
            if arr.ndim == 5:
                batch, frames, h, w, c = arr.shape
                if isinstance(arr, torch.Tensor):
                    arr = arr.reshape(batch * frames, h, w, c)
                else:
                    arr = arr.reshape(batch * frames, h, w, c)
            else:
                # Fallback: take first element
                arr = arr[0]

    # Add batch dimension if 3D
    if arr.ndim == 3:
        arr = (
            arr[None, ...]
            if isinstance(arr, torch.Tensor)
            else np.expand_dims(arr, axis=0)
        )

    return arr


def _replace_seg(seg: Any, **kwargs: Any) -> Any:
    """Replace fields in a SEG namedtuple."""
    if hasattr(seg, "_replace"):
        return seg._replace(**kwargs)

    if hasattr(seg, "_fields"):
        fields = list(seg._fields)
        values = []
        for name in fields:
            values.append(kwargs.get(name, getattr(seg, name)))
        return seg.__class__(*values)

    return seg


class SEGSFixDimensionsNode:
    """
    Fixes SEGS with 5D+ cropped_images by normalizing them to 4D NHWC format.
    Use this AFTER nodes that generate 5D tensors (like DetailerForVideo).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segs": ("SEGS",)}}

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "execute"
    CATEGORY = "link/Impact"
    DESCRIPTION = "Normalizes SEGS cropped_images from 5D+ to 4D NHWC format. Use after DetailerForVideo or other nodes that output 5D tensors."

    def execute(self, segs: tuple):
        header, seg_list = segs
        new_segs = []

        for i, seg in enumerate(seg_list):
            cropped_image = getattr(seg, "cropped_image", None)

            if cropped_image is not None and isinstance(
                cropped_image, (np.ndarray, torch.Tensor)
            ):
                original_shape = cropped_image.shape
                normalized = _normalize_tensor_to_4d(cropped_image)

                if normalized.shape != original_shape:
                    print(
                        f"[SEGS Fix Dimensions] Seg {i}: Normalized from {original_shape} to {normalized.shape}"
                    )

                new_segs.append(_replace_seg(seg, cropped_image=normalized))
            else:
                new_segs.append(seg)

        return ((header, new_segs),)


class SEGSFixCropRegionForNAGNode:
    """
    Fixes SEGS crop_region dimensions to be compatible with Normalized Attention Guidance (NAG).

    NAG requires specific tensor dimension divisibility for attention operations.
    This node ensures crop regions are divisible by 64 (8 for VAE × 8 for attention heads)
    and clears any pre-cached cropped_image data to force fresh cropping with correct dimensions.

    Use this BEFORE DetailerForAnimateDiff when using NAG-patched models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "images": ("IMAGE",),
                "divisor": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "execute"
    CATEGORY = "link/Impact"
    DESCRIPTION = "Fixes SEGS crop_region to be NAG-compatible (divisible by 64 by default). Clears cropped_image cache to prevent dimension mismatches."

    def execute(self, segs: tuple, images: Any, divisor: int = 64):
        header, seg_list = segs
        new_segs = []

        # Get actual image dimensions
        img_height = images.shape[1]
        img_width = images.shape[2]

        for i, seg in enumerate(seg_list):
            crop_region = getattr(seg, "crop_region", None)

            if crop_region is not None:
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

                # Recalculate x2, y2 from x1, y1
                x2 = x1 + width
                y2 = y1 + height

                # If we overflow image bounds, shift the entire region back
                if x2 > img_width:
                    overflow = x2 - img_width
                    x1 = max(0, x1 - overflow)
                    x2 = x1 + width
                    # If still overflowing, reduce width
                    if x2 > img_width:
                        width = img_width - x1
                        width = (width // divisor) * divisor
                        x2 = x1 + width

                if y2 > img_height:
                    overflow = y2 - img_height
                    y1 = max(0, y1 - overflow)
                    y2 = y1 + height
                    # If still overflowing, reduce height
                    if y2 > img_height:
                        height = img_height - y1
                        height = (height // divisor) * divisor
                        y2 = y1 + height

                fixed_crop_region = (x1, y1, x2, y2)

                print(
                    f"[SEGS Fix NAG] Seg {i}: crop_region {crop_region} → {fixed_crop_region} (divisor={divisor})"
                )

                # Clear cropped_image cache to force fresh cropping with correct dimensions
                new_segs.append(
                    _replace_seg(seg, crop_region=fixed_crop_region, cropped_image=None)
                )
            else:
                new_segs.append(seg)

        return ((header, new_segs),)


class DetailerForEachPipeForAnimateDiffFixed:
    """
    Wrapper around DetailerForEachPipeForAnimateDiff that ensures output SEGS are normalized to 4D.

    This fixes the "Expected NHWC tensor, but found 5 dimensions" error that can occur
    when using DetailerForEachPipeForAnimateDiff with certain models and configurations.
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
            },
            "optional": {
                "detailer_hook": ("DETAILER_HOOK",),
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
    DESCRIPTION = "Fixed version of DetailerForEachPipeForAnimateDiff that normalizes SEGS to prevent 5D tensor errors"

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
        refiner_ratio=None,
        detailer_hook=None,
        refiner_basic_pipe_opt=None,
        noise_mask_feather=0,
        scheduler_func_opt=None,
    ):
        """
        Fixed implementation that processes each segment individually with 5D tensor support.
        """
        try:
            from impact.animatediff_nodes import SEGSDetailerForAnimateDiff
        except ImportError:
            raise ImportError(
                "Could not import from Impact Pack. Make sure ComfyUI-Impact-Pack is installed."
            )

        enhanced_segs = []
        cnet_image_list = []

        for sub_seg in segs[1]:
            single_seg = segs[0], [sub_seg]

            # Call the detailer for this single segment
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

            # Normalize the enhanced segment's cropped_image to 4D before pasting
            header, seg_list = enhanced_seg
            normalized_seg_list = []

            for i, seg in enumerate(seg_list):
                cropped_image = getattr(seg, "cropped_image", None)

                if cropped_image is not None and isinstance(
                    cropped_image, (np.ndarray, torch.Tensor)
                ):
                    original_shape = cropped_image.shape
                    normalized = _normalize_tensor_to_4d(cropped_image)

                    if normalized.shape != original_shape:
                        print(
                            f"[DetailerFixed] Seg {i}: Normalized from {original_shape} to {normalized.shape}"
                        )

                    normalized_seg_list.append(
                        _replace_seg(seg, cropped_image=normalized)
                    )
                else:
                    normalized_seg_list.append(seg)

            normalized_enhanced_seg = (header, normalized_seg_list)

            # Use custom 5D-compatible paste
            from .segs_paste_5d import SEGSPaste5D

            image_frames = SEGSPaste5D().doit(
                image_frames, normalized_enhanced_seg, feather, alpha=255
            )[0]

            if cnet_images is not None:
                cnet_image_list.extend(cnet_images)

            if detailer_hook is not None:
                image_frames = detailer_hook.post_paste(image_frames)

            enhanced_segs += normalized_seg_list

        new_segs = segs[0], enhanced_segs
        return (image_frames, new_segs, basic_pipe, cnet_image_list)


__all__ = [
    "SEGSFixDimensionsNode",
    "SEGSFixCropRegionForNAGNode",
    "DetailerForEachPipeForAnimateDiffFixed",
]
