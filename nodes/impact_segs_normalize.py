from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _normalize_cropped_image(image: Any) -> Any:
    if image is None:
        return None

    if isinstance(image, (np.ndarray, torch.Tensor)):
        arr = image
        original_arr = arr
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

        # If still can't normalize to 4D, return the best attempt or original
        if arr.ndim > 4:
            # Try to just take the first element repeatedly until we get to 4D or less
            while arr.ndim > 4:
                arr = arr[0]
            # If we ended up with 3D, add batch dimension
            if arr.ndim == 3:
                arr = arr[None, ...]
            # If still not 4D, return original as last resort
            if arr.ndim != 4:
                return original_arr

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

        for i, seg in enumerate(seg_list):
            original_cropped_image = getattr(seg, "cropped_image", None)
            cropped_mask = getattr(seg, "cropped_mask", None)

            print(
                f"[SEGS Normalize] Seg {i}: original_cropped_image type: {type(original_cropped_image)}"
            )
            if original_cropped_image is not None:
                if isinstance(original_cropped_image, (np.ndarray, torch.Tensor)):
                    print(
                        f"[SEGS Normalize] Seg {i}: original image shape: {original_cropped_image.shape}"
                    )
            if cropped_mask is not None:
                if isinstance(cropped_mask, (np.ndarray, torch.Tensor)):
                    print(f"[SEGS Normalize] Seg {i}: mask shape: {cropped_mask.shape}")

            cropped_image = _normalize_cropped_image(original_cropped_image)

            # Handle dimension mismatch: if image batch size doesn't match mask batch size, replicate the image
            if (
                cropped_image is not None
                and cropped_mask is not None
                and isinstance(cropped_image, (np.ndarray, torch.Tensor))
                and isinstance(cropped_mask, (np.ndarray, torch.Tensor))
            ):
                image_batch = len(cropped_image)
                mask_batch = len(cropped_mask)

                if image_batch != mask_batch:
                    print(
                        f"[SEGS Normalize] Seg {i}: Batch mismatch! Image: {image_batch}, Mask: {mask_batch}"
                    )
                    # Replicate the image to match mask batch size
                    if image_batch == 1 and mask_batch > 1:
                        # Convert to torch if numpy
                        if isinstance(cropped_image, np.ndarray):
                            cropped_image = torch.from_numpy(cropped_image)
                        # Repeat the single image for all frames
                        cropped_image = cropped_image.repeat(mask_batch, 1, 1, 1)
                        print(
                            f"[SEGS Normalize] Seg {i}: Replicated image to shape: {cropped_image.shape}"
                        )

            if cropped_image is not None:
                if isinstance(cropped_image, (np.ndarray, torch.Tensor)):
                    print(
                        f"[SEGS Normalize] Seg {i}: final normalized shape: {cropped_image.shape}"
                    )
            else:
                print(f"[SEGS Normalize] Seg {i}: normalized to None!")

            new_segs.append(_replace_seg(seg, cropped_image=cropped_image))

        return ((header, new_segs),)


__all__ = ["SEGSNormalizeForAnimateDiffNode"]
