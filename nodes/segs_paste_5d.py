"""5D-compatible SEGSPaste node for AnimateDiff workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class SEGSPaste5D:
    """
    5D-compatible version of SEGSPaste that handles video frame batches.

    This node properly handles cropped_images with 5D tensors (batch, frames, H, W, C)
    which can occur when using DetailerForAnimateDiff nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
            },
            "optional": {
                "ref_image_opt": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"
    CATEGORY = "link/Impact"
    DESCRIPTION = "5D-compatible SEGSPaste for AnimateDiff video workflows. Handles multi-frame cropped_images."

    @staticmethod
    def _normalize_to_4d(tensor: Any) -> Any:
        """Normalize tensor to 4D NHWC format."""
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
                # If 5D with no singleton dims, assume [batch, frames, H, W, C]
                # Flatten to [batch*frames, H, W, C]
                if arr.ndim == 5:
                    b, f, h, w, c = arr.shape
                    if isinstance(arr, torch.Tensor):
                        arr = arr.reshape(b * f, h, w, c)
                    else:
                        arr = arr.reshape(b * f, h, w, c)
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

    def doit(
        self,
        image: torch.Tensor,
        segs: tuple,
        feather: int,
        alpha: int = 255,
        ref_image_opt: torch.Tensor | None = None,
    ):
        """Paste enhanced segments back into image frames with 5D tensor support."""
        try:
            # Import Impact Pack utilities
            from impact import utils
            from impact.utils import tensor_gaussian_blur_mask
        except ImportError:
            raise ImportError(
                "Could not import from Impact Pack. Make sure ComfyUI-Impact-Pack is installed."
            )

        if len(segs[1]) == 0:
            return (image,)

        batch_size = image.shape[0]
        result = torch.empty_like(image)

        with torch.no_grad():
            for i in range(batch_size):
                image_i = image[i].unsqueeze(0).clone()

                for seg in segs[1]:
                    ref_image = None

                    # Handle cropped_image from segment
                    if ref_image_opt is None and seg.cropped_image is not None:
                        cropped_image = seg.cropped_image

                        # Convert to tensor if numpy
                        if isinstance(cropped_image, np.ndarray):
                            cropped_image = torch.from_numpy(cropped_image)

                        # Normalize 5D to 4D
                        cropped_image = self._normalize_to_4d(cropped_image)

                        # Safely index by frame
                        if cropped_image is not None and i < len(cropped_image):
                            ref_image = cropped_image[i].unsqueeze(0)
                        elif cropped_image is not None:
                            # Use last frame if index out of bounds
                            ref_image = cropped_image[-1].unsqueeze(0)

                    elif ref_image_opt is not None:
                        ref_tensor = ref_image_opt[i].unsqueeze(0)
                        ref_image = utils.crop_image(ref_tensor, seg.crop_region)

                    if ref_image is None:
                        continue

                    # Handle mask
                    cmask = seg.cropped_mask
                    if cmask.ndim == 3 and len(cmask) == batch_size:
                        mask = cmask[i]
                    elif cmask.ndim == 3 and len(cmask) > 1:
                        # Combine multiple masks
                        mask = torch.any(cmask > 0.1, dim=0).float()
                    else:  # ndim == 2
                        mask = cmask

                    # Apply feathering and alpha
                    mask = tensor_gaussian_blur_mask(mask, feather) * (alpha / 255.0)

                    # Ensure same device
                    mask = mask.to(image_i.device)
                    ref_image = ref_image.to(image_i.device)

                    # Paste
                    x, y, *_ = seg.crop_region
                    utils.tensor_paste(image_i, ref_image, (x, y), mask)

                result[i] = image_i[0]

        return (result,)


__all__ = ["SEGSPaste5D"]
