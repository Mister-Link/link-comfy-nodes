"""SEGS fixer nodes to handle dimension issues with Impact Pack nodes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _normalize_tensor_to_4d(tensor: Any) -> Any:
    """Normalize a tensor to 4D NHWC format."""
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


class SEGSEnsureCroppedImageNode:
    def execute(self, segs: tuple, images: Any):
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

                # Ensure dimensions are divisible by 8 for VAE
                width = x2 - x1
                height = y2 - y1

                width = (width // 8) * 8
                height = (height // 8) * 8

                x2 = x1 + width
                y2 = y1 + height

                fixed_crop_region = (x1, y1, x2, y2)

                print(
                    f"[SEGS Ensure Cropped] Seg {i}: Fixed crop_region from {crop_region} to {fixed_crop_region}"
                )

                # DO NOT create cropped_image - let AnimateDiff handle it
                new_segs.append(
                    _replace_seg(seg, crop_region=fixed_crop_region, cropped_image=None)
                )
            else:
                new_segs.append(seg)

        return ((header, new_segs),)


__all__ = ["SEGSFixDimensionsNode", "SEGSEnsureCroppedImageNode"]
