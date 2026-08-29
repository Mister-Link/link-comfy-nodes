from __future__ import annotations

import cv2
import numpy as np
import torch


class StabilizeSpriteSequenceNode:
    """Scale a sprite sequence consistently, then tightly crop each frame.

    Follows stabilize_poc3.py: Otsu mask polarity detection, largest connected
    component cleanup, smoothed height/width for scale only, and a fresh alpha
    bbox after scaling. ComfyUI IMAGE batches need one shape, so each tight
    crop is fitted into the requested output box without stretching.
    """

    SMOOTH_WINDOW = 5
    MARGIN_FRAC = 0.05
    FILL_MARGIN = 0.98

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "stabilize"
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "pixel_width": ("INT", {"default": 70, "min": 1, "max": 16384, "step": 1}),
            "pixel_height": ("INT", {"default": 160, "min": 1, "max": 16384, "step": 1}),
            "upscaled_width": ("INT", {"default": 512, "min": 1, "max": 65536, "step": 1}),
            "upscaled_height": ("INT", {"default": 1152, "min": 1, "max": 65536, "step": 1}),
        }}

    @staticmethod
    def _clean_foreground(mask):
        mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        _, otsu_bin = cv2.threshold(mask_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border = np.concatenate((otsu_bin[0], otsu_bin[-1], otsu_bin[:, 0], otsu_bin[:, -1]))
        bg_is_255 = np.median(border) > 127
        foreground = (otsu_bin == 0).astype(np.uint8) if bg_is_255 else (otsu_bin == 255).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        if num_labels <= 1:
            h, w = mask.shape
            return (0, 0, w, h), 1.0 if bg_is_255 else 0.0, np.ones_like(mask, dtype=np.float32)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, _ = stats[largest]
        component = (labels == largest).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        keep = cv2.dilate(component, kernel, iterations=1).astype(np.float32)
        return (x, y, x + w, y + h), 1.0 if bg_is_255 else 0.0, keep

    @classmethod
    def _smooth(cls, signal):
        pad = cls.SMOOTH_WINDOW // 2
        return np.convolve(np.pad(signal, pad, mode="edge"),
                           np.ones(cls.SMOOTH_WINDOW) / cls.SMOOTH_WINDOW, mode="valid")

    def stabilize(self, image, mask, pixel_width, pixel_height, upscaled_width, upscaled_height):
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

        n, input_h, input_w, channels = images.shape
        ref_w, ref_h = int(upscaled_width), int(upscaled_height)
        output_w, output_h = int(pixel_width), int(pixel_height)
        raw_height, raw_width = np.zeros(n), np.zeros(n)
        keep_regions, bg_values = [], []

        for i in range(n):
            frame_mask = masks[0 if masks.shape[0] == 1 else i]
            (x1, y1, x2, y2), bg_value, keep = self._clean_foreground(frame_mask)
            raw_height[i] = max(1, y2 - y1)
            raw_width[i] = max(1, x2 - x1)
            keep_regions.append(keep)
            bg_values.append(bg_value)

        smooth_height = self._smooth(raw_height)
        smooth_width = self._smooth(raw_width)
        target_height = float(ref_h)
        for width, height in zip(smooth_width, smooth_height):
            target_height = min(target_height, ref_w / (width / height))
        target_height *= self.FILL_MARGIN

        out_images, out_masks = [], []
        for i in range(n):
            frame = images[i]
            frame_mask = masks[0 if masks.shape[0] == 1 else i]
            alpha = (1.0 - frame_mask if bg_values[i] > 0.5 else frame_mask) * keep_regions[i]
            scale = target_height / max(smooth_height[i], 1.0)
            scaled_w = max(1, round(input_w * scale))
            scaled_h = max(1, round(input_h * scale))
            scaled_image = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            scaled_alpha = cv2.resize(alpha, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            ys, xs = np.where(scaled_alpha > 0.02)
            if ys.size == 0:
                crop = np.zeros((1, 1, channels + 1), dtype=np.float32)
            else:
                x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                content_w, content_h = x2 - x1, y2 - y1
                mx, my = round(content_w * self.MARGIN_FRAC), round(content_h * self.MARGIN_FRAC)
                crop = np.dstack((scaled_image[y1:y2, x1:x2], scaled_alpha[y1:y2, x1:x2]))
                crop = cv2.copyMakeBorder(crop, my, my, mx, mx, cv2.BORDER_CONSTANT, value=0)

            fit = min(output_w / crop.shape[1], output_h / crop.shape[0])
            fit_w, fit_h = max(1, round(crop.shape[1] * fit)), max(1, round(crop.shape[0] * fit))
            resized = cv2.resize(crop, (fit_w, fit_h), interpolation=cv2.INTER_LANCZOS4)
            canvas = np.zeros((output_h, output_w, channels + 1), dtype=np.float32)
            dx, dy = (output_w - fit_w) // 2, (output_h - fit_h) // 2
            canvas[dy:dy + fit_h, dx:dx + fit_w] = resized
            out_images.append(canvas[..., :channels])
            out_masks.append(canvas[..., -1])

        return (torch.from_numpy(np.stack(out_images)).to(image.device).clamp(0.0, 1.0),
                torch.from_numpy(np.stack(out_masks)).to(mask.device).clamp(0.0, 1.0))
