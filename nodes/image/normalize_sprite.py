from __future__ import annotations

import cv2
import numpy as np
import torch


class NormalizeSpriteEntityHeightNode:
    """Fit cropped frames into the calculator's upscaled canvas, then reduce them."""

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
                "pixel_width": ("INT", {"default": 70, "min": 1, "max": 16384, "step": 1}),
                "pixel_height": ("INT", {"default": 160, "min": 1, "max": 16384, "step": 1}),
                "upscaled_width": ("INT", {"default": 512, "min": 1, "max": 65536, "step": 1}),
                "upscaled_height": ("INT", {"default": 1152, "min": 1, "max": 65536, "step": 1}),
                "anchor": (["center", "bottom_center", "top_center"], {"default": "center"}),
            }
        }

    @staticmethod
    def _offset(canvas_w, canvas_h, content_w, content_h, anchor):
        x = (canvas_w - content_w) // 2
        if anchor == "bottom_center":
            y = canvas_h - content_h
        elif anchor == "top_center":
            y = 0
        else:
            y = (canvas_h - content_h) // 2
        return max(0, x), max(0, y)

    @staticmethod
    def _mask_bbox(mask_frame, threshold=0.02):
        ys, xs = np.where(mask_frame > threshold)
        if ys.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def normalize(self, image, mask, pixel_width, pixel_height, upscaled_width, upscaled_height, anchor="center"):
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

        canvas_w, canvas_h = int(upscaled_width), int(upscaled_height)
        output_w, output_h = int(pixel_width), int(pixel_height)

        out_images, out_masks = [], []
        for i, frame in enumerate(images):
            frame_mask = masks[0 if masks.shape[0] == 1 else i]

            # Crop to this frame's own alpha content first -- each frame in a
            # sequence can have a different amount of transparent padding
            # around the entity (e.g. a jump pose rendered smaller within
            # the same canvas), so a bbox shared across the batch (as
            # AutoCropper produces) doesn't equalize apparent entity size.
            # Fitting per-frame content into the canvas, rather than the
            # whole per-frame image, is what actually normalizes height.
            bbox = self._mask_bbox(frame_mask)
            if bbox is None:
                content, content_mask = frame, frame_mask
            else:
                x1, y1, x2, y2 = bbox
                content, content_mask = frame[y1:y2, x1:x2], frame_mask[y1:y2, x1:x2]

            content_h, content_w = content.shape[:2]
            fit = min(canvas_w / content_w, canvas_h / content_h)
            fitted_w = max(1, min(canvas_w, round(content_w * fit)))
            fitted_h = max(1, min(canvas_h, round(content_h * fit)))
            offset_x, offset_y = self._offset(canvas_w, canvas_h, fitted_w, fitted_h, anchor)

            fitted = cv2.resize(content, (fitted_w, fitted_h), interpolation=cv2.INTER_LANCZOS4)
            fitted_mask = cv2.resize(content_mask, (fitted_w, fitted_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((canvas_h, canvas_w, frame.shape[2]), dtype=np.float32)
            canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            canvas[offset_y:offset_y + fitted_h, offset_x:offset_x + fitted_w] = fitted
            canvas_mask[offset_y:offset_y + fitted_h, offset_x:offset_x + fitted_w] = fitted_mask
            out_images.append(cv2.resize(canvas, (output_w, output_h), interpolation=cv2.INTER_AREA))
            out_masks.append(cv2.resize(canvas_mask, (output_w, output_h), interpolation=cv2.INTER_AREA))

        output_images = torch.from_numpy(np.stack(out_images)).to(image.device).clamp(0.0, 1.0)
        output_masks = torch.from_numpy(np.stack(out_masks)).to(mask.device).clamp(0.0, 1.0)
        return output_images, output_masks
