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
    """
    Ensures SEGS have cropped_image data by cropping from provided images if missing.
    Use this BEFORE nodes that require cropped_image (like SEGSPreview without fallback).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "execute"
    CATEGORY = "link/Impact"
    DESCRIPTION = "Ensures SEGS have cropped_image by cropping from images if missing. Use before DetailerForVideo."

    def execute(self, segs: tuple, images: Any):
        # Import Impact Pack utils dynamically
        import sys

        impact_path = "/workspace/ComfyUI/custom_nodes/comfyui-impact-pack/modules"
        if impact_path not in sys.path:
            sys.path.insert(0, impact_path)

        try:
            from impact import utils as impact_utils
        except ImportError:
            # Fallback for local development
            impact_path_local = (
                "/home/developer/ComfyUI/custom_nodes/comfyui-impact-pack/modules"
            )
            if impact_path_local not in sys.path:
                sys.path.insert(0, impact_path_local)
            from impact import utils as impact_utils

        header, seg_list = segs
        new_segs = []

        for i, seg in enumerate(seg_list):
            cropped_image = getattr(seg, "cropped_image", None)

            if cropped_image is None:
                crop_region = getattr(seg, "crop_region", None)
                if crop_region is not None:
                    print(
                        f"[SEGS Ensure Cropped] Seg {i}: Creating cropped_image from images"
                    )

                    # Crop from each frame
                    cropped_frames = None
                    for frame in images:
                        frame_unsqueezed = frame.unsqueeze(0)
                        cropped = impact_utils.crop_tensor4(
                            frame_unsqueezed, crop_region
                        )
                        if cropped_frames is None:
                            cropped_frames = cropped
                        else:
                            cropped_frames = torch.cat((cropped_frames, cropped), dim=0)

                    print(
                        f"[SEGS Ensure Cropped] Seg {i}: Created cropped_image with shape {cropped_frames.shape}"
                    )
                    new_segs.append(_replace_seg(seg, cropped_image=cropped_frames))
                else:
                    new_segs.append(seg)
            else:
                new_segs.append(seg)

        return ((header, new_segs),)


__all__ = ["SEGSFixDimensionsNode", "SEGSEnsureCroppedImageNode"]
