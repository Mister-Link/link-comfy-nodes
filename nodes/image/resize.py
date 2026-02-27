from __future__ import annotations

import torch.nn.functional as F


class ResizeImageAndMaskBySideNode:
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask")
    FUNCTION = "resize_by_side"
    CATEGORY = "image/transform"

    _INTERPOLATION_MODES = {
        "bicubic": "bicubic",
        "bilinear": "bilinear",
        "nearest exact": "nearest-exact",
    }
    _SIDES = ["longer", "shorter"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "length": (
                    "INT",
                    {"default": 512, "min": 1, "max": 16384, "step": 1},
                ),
                "interpolation": (list(cls._INTERPOLATION_MODES.keys()),),
                "side": (cls._SIDES,),
            }
        }

    def resize_by_side(self, image, length: int, mask, interpolation: str, side: str):
        images = image.detach().float()
        if images.ndim != 4:
            raise ValueError("Expected image with shape (N, H, W, C)")

        height, width = images.shape[1], images.shape[2]

        mask = mask.detach().float()
        if mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 3:
            raise ValueError("Expected mask with shape (N, H, W)")
        if mask.shape[0] != images.shape[0]:
            raise ValueError("Mask batch size does not match images")
        if mask.shape[1] != height or mask.shape[2] != width:
            raise ValueError("Mask dimensions do not match image dimensions")

        if side == "longer":
            if width >= height:
                target_width = length
                target_height = max(1, int(round(height * (length / width))))
            else:
                target_height = length
                target_width = max(1, int(round(width * (length / height))))
        else:
            if width <= height:
                target_width = length
                target_height = max(1, int(round(height * (length / width))))
            else:
                target_height = length
                target_width = max(1, int(round(width * (length / height))))

        images_chw = images.permute(0, 3, 1, 2)
        mask_chw = mask.unsqueeze(1)

        mode = self._INTERPOLATION_MODES[interpolation]
        align_corners = (
            False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None
        )
        interpolate_kwargs = {}
        if align_corners is not None:
            interpolate_kwargs["align_corners"] = align_corners

        resized_images = F.interpolate(
            images_chw,
            size=(target_height, target_width),
            mode=mode,
            antialias=False,
            **interpolate_kwargs,
        )
        resized_mask = F.interpolate(
            mask_chw,
            size=(target_height, target_width),
            mode=mode,
            antialias=False,
            **interpolate_kwargs,
        )

        result_images = resized_images.permute(0, 2, 3, 1)
        result_mask = resized_mask[:, 0]
        return (result_images, result_mask)
