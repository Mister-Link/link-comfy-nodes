from __future__ import annotations

import cv2
import numpy as np
import torch


class StabilizeSpriteSequenceNode:
    """Normalize entity size and ground position across a frame sequence.

    Fully automatic -- no tunables. Three things make this work without any
    knobs to fiddle with:

      1. Mask polarity is auto-detected per frame by sampling the border
         (whichever value dominates the border is background, regardless of
         whether that's 0 or 1). Assuming a fixed polarity broke badly on a
         mask where the background was 1 and the subject 0 -- the "content"
         bbox came out as basically the whole canvas, so nothing actually
         got cropped tighter.
      2. target_height is derived automatically: the tightest height that
         fills the canvas as much as possible without clipping any frame's
         (temporally smoothed) width against the canvas width.
      3. Per-frame alpha-bbox detection is noisy on its own (a stray hair
         wisp or antialiased fringe pixel can swing the detected bbox by
         several pixels frame to frame), and cropping to an integer pixel
         box before resizing compounds that with rounding jitter. This
         tracks each frame's entity center/ground/height across the whole
         sequence, smooths those three signals temporally to remove
         per-frame detection noise, then warps each frame onto the canvas
         with a single subpixel-accurate affine transform (scale +
         translate) instead of a discrete crop-then-resize.
    """

    SMOOTH_WINDOW = 5
    FILL_MARGIN = 0.98
    ANCHOR_X_FRAC = 0.5
    ANCHOR_Y_FRAC = 0.98

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
            }
        }

    @staticmethod
    def _foreground_bbox(mask_frame):
        h, w = mask_frame.shape
        border = np.concatenate(
            [mask_frame[0, :], mask_frame[-1, :], mask_frame[:, 0], mask_frame[:, -1]]
        )
        bg_value = float(np.median(border))
        fg = (np.abs(mask_frame - bg_value) > 0.5).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if num_labels <= 1:
            return None, bg_value
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w2, h2, _ = stats[largest]
        return (x, y, x + w2, y + h2), bg_value

    @classmethod
    def _smooth(cls, signal):
        window = cls.SMOOTH_WINDOW
        if window <= 1:
            return signal
        pad = window // 2
        padded = np.pad(signal, pad, mode="edge")
        kernel = np.ones(window) / window
        return np.convolve(padded, kernel, mode="valid")

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

        num_frames, _, _, num_channels = images.shape
        canvas_w, canvas_h = int(upscaled_width), int(upscaled_height)
        output_w, output_h = int(pixel_width), int(pixel_height)

        raw_center_x = np.zeros(num_frames, dtype=np.float64)
        raw_ground_y = np.zeros(num_frames, dtype=np.float64)
        raw_height = np.zeros(num_frames, dtype=np.float64)
        raw_width = np.zeros(num_frames, dtype=np.float64)
        bg_values = np.zeros(num_frames, dtype=np.float64)

        for i in range(num_frames):
            frame_mask = masks[0 if masks.shape[0] == 1 else i]
            bbox, bg_value = self._foreground_bbox(frame_mask)
            if bbox is None:
                fh, fw = frame_mask.shape
                x1, y1, x2, y2 = 0, 0, fw, fh
            else:
                x1, y1, x2, y2 = bbox
            raw_center_x[i] = (x1 + x2) / 2.0
            raw_ground_y[i] = y2
            raw_height[i] = max(1, y2 - y1)
            raw_width[i] = max(1, x2 - x1)
            bg_values[i] = bg_value

        smooth_center_x = self._smooth(raw_center_x)
        smooth_ground_y = self._smooth(raw_ground_y)
        smooth_height = self._smooth(raw_height)
        smooth_width = self._smooth(raw_width)

        # Tightest target_height that fills the canvas as much as possible
        # without ever clipping any frame's (smoothed) width against canvas_w.
        max_target_height = float(canvas_h)
        for i in range(num_frames):
            aspect = smooth_width[i] / smooth_height[i]
            max_target_height = min(max_target_height, canvas_w / aspect)
        target_height = max_target_height * self.FILL_MARGIN

        anchor_x = canvas_w * self.ANCHOR_X_FRAC
        anchor_y = canvas_h * self.ANCHOR_Y_FRAC

        out_images, out_masks = [], []
        for i in range(num_frames):
            frame = images[i]
            frame_mask = masks[0 if masks.shape[0] == 1 else i]
            bg_value = bg_values[0 if masks.shape[0] == 1 else i]

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
                borderValue=(0.0,) * num_channels,
            )
            warped_mask = cv2.warpAffine(
                frame_mask,
                transform,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=float(bg_value),
            )

            out_images.append(cv2.resize(warped_frame, (output_w, output_h), interpolation=cv2.INTER_AREA))
            out_masks.append(cv2.resize(warped_mask, (output_w, output_h), interpolation=cv2.INTER_AREA))

        output_images = torch.from_numpy(np.stack(out_images)).to(image.device).clamp(0.0, 1.0)
        output_masks = torch.from_numpy(np.stack(out_masks)).to(mask.device).clamp(0.0, 1.0)
        return output_images, output_masks
