from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _normalize_cropped_image(image: Any) -> Any:
    if image is None:
        return None

    if isinstance(image, (np.ndarray, torch.Tensor)):
        arr = image
        # Squeeze singleton dims until we reach 4D if possible.
        while arr.ndim > 4:
            squeeze_dim = None
            for dim, size in enumerate(arr.shape):
                if size == 1:
                    squeeze_dim = dim
                    break
            if squeeze_dim is None:
                arr = arr[0]
            else:
                arr = arr.squeeze(squeeze_dim)

        if arr.ndim == 3:
            arr = arr[None, ...]

        if arr.ndim > 4:
            return None

        return arr

    return image


def _replace_seg(seg: Any, **kwargs: Any) -> Any:
    if hasattr(seg, "_replace"):
        return seg._replace(**kwargs)

    if hasattr(seg, "_fields"):
        fields = list(seg._fields)
        values = []
        for name in fields:
            values.append(kwargs.get(name, getattr(seg, name)))
        return seg.__class__(*values)

    return seg


class SEGSNormalizeForAnimateDiffNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segs": ("SEGS",)}}

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "execute"
    CATEGORY = "link/Impact"

    def execute(self, segs: tuple):
        header, seg_list = segs
        new_segs = []










                and and ))
            )






                            for seg in seg_list:
            cropped_image = _normalize_cropped_image(
                getattr(seg, "cropped_image", None)
            )
            new_segs.append(_replace_seg(seg, cropped_image=cropped_image))

        return ((header, new_segs),)


class SEGSNormalizeBeforeDetailerNode:
    """
    Normalizes SEGS before passing to Detailer by ensuring cropped_image exists.
    If cropped_image is None, crops from the provided image_frames.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "image_frames": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "execute"
    CATEGORY = "link/Impact"

    def execute(self, segs: tuple, image_frames: Any):
        import sys
        sys.path.insert(0, "/home/developer/ComfyUI/custom_nodes/comfyui-impact-pack/modules")
        from impact import utils as impact_utils

        header, seg_list = segs
        new_segs = []

        print(f"[SEGS Normalize Before Detailer] Processing {len(seg_list)} segments")

        for i, seg in enumerate(seg_list):
            cropped_image = getattr(seg, "cropped_image", None)

            # If no cropped_image, create it by cropping from image_frames
            if cropped_image is None:
                print(f"[SEGS Normalize Before Detailer] Seg {i}: Creating cropped_image from image_frames")
                crop_region = getattr(seg, "crop_region", None)
                if crop_region is not None:
                    # Crop from each frame
                    cropped_frames = None
                    for frame_idx, frame in enumerate(image_frames):
                        frame = frame.unsqueeze(0)
                        cropped = impact_utils.crop_tensor4(frame, crop_region)
                        if cropped_frames is None:
                            cropped_frames = cropped
                        else:
                            cropped_frames = torch.cat((cropped_frames, cropped), dim=0)
                    cropped_image = cropped_frames
                    print(f"[SEGS Normalize Before Detailer] Seg {i}: Created cropped_image with shape {cropped_image.shape}")
            else:
                print(f"[SEGS Normalize Before Detailer] Seg {i}: cropped_image already exists with shape {cropped_image.shape}")

            new_segs.append(_replace_seg(seg, cropped_image=cropped_image))

        return ((header, new_segs),)


__all__ = ["SEGSNormalizeForAnimateDiffNode", "SEGSNormalizeBeforeDetailerNode"]
