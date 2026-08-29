from __future__ import annotations

import cv2
import numpy as np
import torch


class StabilizeSpriteSequenceNode:
    """Normalize entity size and ground position across a frame sequence.

    Per-frame alpha-bbox detection is noisy on its own (a stray hair wisp or
    antialiased fringe pixel can swing the detected bbox by several pixels
    frame to frame), and cropping to an integer pixel box before resizing
    compounds that with rounding jitter. This avoids both: it tracks each
    frame's entity center/ground/height across the whole sequence, smooths
    those three signals temporally to remove per-frame detection noise, then
    warps each frame onto the canvas with a single subpixel-accurate affine
    transform (scale + translate) instead of a discrete crop-then-resize.
    """

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "stabilize"
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
                "target_height_frac": (
                    "FLOAT",
                    {
                        "default": 0.78,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Entity height as a fraction of upscaled_height after normalization.",
                    },
                ),
                "anchor_x_frac": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "anchor_y_frac": (
                    "FLOAT",
                    {
                        "default": 0.92,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Where the entity's ground contact point lands, as a fraction down the canvas.",
                    },
                ),
                "smooth_window": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 63,
                        "step": 2,
                        "tooltip": "Frames averaged together when tracking center/ground/height. Larger = smoother but laggier.",
                    },
                ),
                "alpha_threshold": (
                    "FLOAT",
                    {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    @staticmethod
    def _largest_component_bbox(mask_frame, threshold):
        mask_u8 = (mask_frame > threshold).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels <= 1:
            return None
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, _ = stats[largest]
        return x, y, x + w, y + h

    @staticmethod
    def _smooth(signal, window):
        if window <= 1:
            return signal
        pad = window // 2
        padded = np.pad(signal, pad, mode="edge")
        kernel = np.ones(window) / window
        return np.convolve(padded, kernel, mode="valid")

    def stabilize(
        self,
        image,
        mask,
        pixel_width,
        pixel_height,
        upscaled_width,
        upscaled_height,
        target_height_frac=0.78,
        anchor_x_frac=0.5,
        anchor_y_frac=0.92,
        smooth_window=5,
        alpha_threshold=0.04,
    ):
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

        num_frames, _, _, num_channels = images.shape
        canvas_w, canvas_h = int(upscaled_width), int(upscaled_height)
        output_w, output_h = int(pixel_width), int(pixel_height)
        target_height = canvas_h * float(target_height_frac)
        anchor_x = canvas_w * float(anchor_x_frac)
        anchor_y = canvas_h * float(anchor_y_frac)

        raw_center_x = np.zeros(num_frames, dtype=np.float64)
        raw_ground_y = np.zeros(num_frames, dtype=np.float64)
        raw_height = np.zeros(num_frames, dtype=np.float64)

        for i in range(num_frames):
            frame_mask = masks[0 if masks.shape[0] == 1 else i]
            bbox = self._largest_component_bbox(frame_mask, alpha_threshold)
            if bbox is None:
                h, w = frame_mask.shape
                x1, y1, x2, y2 = 0, 0, w, h
            else:
                x1, y1, x2, y2 = bbox
            raw_center_x[i] = (x1 + x2) / 2.0
            raw_ground_y[i] = y2
            raw_height[i] = max(1, y2 - y1)

        window = max(1, int(smooth_window))
        smooth_center_x = self._smooth(raw_center_x, window)
        smooth_ground_y = self._smooth(raw_ground_y, window)
        smooth_height = self._smooth(raw_height, window)

        border_value = (0.0,) * num_channels

        out_images, out_masks = [], []
        for i in range(num_frames):
            frame = images[i]
            frame_mask = masks[0 if masks.shape[0] == 1 else i]

            scale = target_height / smooth_height[i]
            tx = anchor_x - smooth_center_x[i] * scale
            ty = anchor_y - smooth_ground_y[i] * scale
            transform = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)

            warped_frame = cv2.warpAffine(
                frame,
                transform,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )
            warped_mask = cv2.warpAffine(
                frame_mask,
                transform,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

            out_images.append(cv2.resize(warped_frame, (output_w, output_h), interpolation=cv2.INTER_AREA))
            out_masks.append(cv2.resize(warped_mask, (output_w, output_h), interpolation=cv2.INTER_AREA))

        output_images = torch.from_numpy(np.stack(out_images)).to(image.device).clamp(0.0, 1.0)
        output_masks = torch.from_numpy(np.stack(out_masks)).to(mask.device).clamp(0.0, 1.0)
        return output_images, output_masks
