from __future__ import annotations

import cv2
import numpy as np
import torch


class NormalizeSpriteEntityHeightNode:
    """Scale masked entities to a standard height while preserving frame canvases."""

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "normalize"
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "pixel_height": ("INT", {"default": 160, "min": 1, "max": 16384, "step": 1}),
                "anchor": (["center", "bottom_center", "top_center"], {"default": "center"}),
            }
        }

    @staticmethod
    def _bounds(mask: np.ndarray):
        ys, xs = np.where(mask > 0.5)
        if xs.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @staticmethod
    def _anchor(bounds, anchor):
        x0, y0, x1, y1 = bounds
        if anchor == "bottom_center":
            return (0.5 * (x0 + x1 - 1), float(y1 - 1))
        if anchor == "top_center":
            return (0.5 * (x0 + x1 - 1), float(y0))
        return (0.5 * (x0 + x1 - 1), 0.5 * (y0 + y1 - 1))

    def normalize(self, image, mask, pixel_height, anchor="center"):
        images = image.detach().cpu().numpy().astype(np.float32)
        masks = mask.detach().cpu().numpy().astype(np.float32)
        if masks.ndim == 4 and masks.shape[-1] == 1:
            masks = masks[..., 0]
        if images.ndim != 4 or masks.ndim != 3:
            raise ValueError("Expected IMAGE (N,H,W,C) and MASK (N,H,W).")
        if masks.shape[0] not in (1, images.shape[0]):
            raise ValueError("Mask batch must contain one mask or match the image batch.")
        if masks.shape[1:] != images.shape[1:3]:
            raise ValueError("Mask dimensions must match image dimensions.")

        out_images, out_masks = [], []
        for i, frame in enumerate(images):
            frame_mask = masks[0 if masks.shape[0] == 1 else i]
            bounds = self._bounds(frame_mask)
            if bounds is None:
                out_images.append(frame.copy())
                out_masks.append(frame_mask.copy())
                continue
            x0, y0, x1, y1 = bounds
            scale = float(pixel_height) / max(1, y1 - y0)
            ax, ay = self._anchor(bounds, anchor)
            matrix = np.array(
                [[scale, 0.0, ax - scale * ax], [0.0, scale, ay - scale * ay]],
                dtype=np.float32,
            )
            h, w = frame.shape[:2]
            resized = cv2.warpAffine(
                frame, matrix, (w, h), flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            resized_mask = cv2.warpAffine(
                frame_mask, matrix, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            out_images.append(resized)
            out_masks.append(np.clip(resized_mask, 0.0, 1.0))

        output_images = torch.from_numpy(np.stack(out_images)).to(image.device).clamp(0.0, 1.0)
        output_masks = torch.from_numpy(np.stack(out_masks)).to(mask.device).clamp(0.0, 1.0)
        return output_images, output_masks
