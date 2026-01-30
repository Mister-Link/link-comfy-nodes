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


__all__ = ["SEGSFixDimensionsNode", "SEGSFixCropRegionForNAGNode"]
