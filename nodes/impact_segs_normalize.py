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

        for seg in seg_list:
            cropped_image = _normalize_cropped_image(getattr(seg, "cropped_image", None))
            new_segs.append(_replace_seg(seg, cropped_image=cropped_image))

        return ((header, new_segs),)


__all__ = ["SEGSNormalizeForAnimateDiffNode"]
