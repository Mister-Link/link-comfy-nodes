"""AutoCropper node for automatic content-aware cropping using anime segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import folder_paths

# Add Video Cropper to path
_VIDEO_CROPPER_PATH = Path(__file__).parent.parent / "Video Cropper"
if str(_VIDEO_CROPPER_PATH) not in sys.path:
    sys.path.insert(0, str(_VIDEO_CROPPER_PATH))

try:
    from model_utils import load_anime_seg_model

    ANIME_SEG_AVAILABLE = True
except Exception as e:
    ANIME_SEG_AVAILABLE = False
    print(f"[AutoCropper] AnimeSegmentation not available: {e}")


class AutoCropperNode:
    """Automatically crop frames to content bounds using anime segmentation model."""

    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("cropped_frames", "cropped_alpha", "width", "height")
    FUNCTION = "auto_crop"

    _model = None
    _device = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "method": (
                    [
                        "anime_seg",
                        "alpha_channel",
                        "bbox_detection",
                        "contour_detection",
                    ],
                    {"default": "bbox_detection"},
                ),
                "sensitivity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "padding": (
                    "INT",
                    {"default": 5, "min": 0, "max": 500, "step": 1},
                ),
            },
            "optional": {
                "alpha": ("MASK",),
            },
        }

    @classmethod
    def _get_model(cls):
        """Lazy load the segmentation model with fallback support."""
        if not ANIME_SEG_AVAILABLE:
            raise RuntimeError(
                "AnimeSegmentation model not available. "
                "Please ensure the Video Cropper folder is properly set up."
            )

        if cls._model is None:
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model = load_anime_seg_model(folder_paths, cls._device)

        return cls._model, cls._device

    @torch.no_grad()
    def _segment_frame_anime(self, frame_np, model, device, sensitivity):
        """Segment a single frame using anime segmentation model."""
        if frame_np.shape[2] == 4:
            bgr = frame_np[:, :, :3]
        else:
            bgr = frame_np

        bgr_cv2 = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        rgb = cv2.cvtColor(bgr_cv2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        pred = model(tensor)[0, 0].cpu().numpy()
        mask = (pred > sensitivity).astype(np.uint8) * 255

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest).astype(np.uint8) * 255

        return mask

    def _detect_bbox_from_pixels(self, frame_np, sensitivity):
        """Detect bounding box from non-black pixels, ignoring transparency."""
        # Convert to uint8 for processing
        frame_uint8 = (frame_np * 255).astype(np.uint8)

        # Create mask from non-black pixels in RGB channels only
        # Ignore alpha channel completely
        if frame_np.shape[2] == 4:
            # Only check RGB channels, ignore alpha
            rgb_channels = frame_uint8[:, :, :3]
        else:
            rgb_channels = frame_uint8

        # Check if any RGB channel is above threshold
        mask = (
            np.any(rgb_channels > int(sensitivity * 255), axis=2).astype(np.uint8) * 255
        )

        return mask

    def _detect_contour(self, frame_np, sensitivity):
        """Detect largest contour in the frame."""
        frame_uint8 = (frame_np * 255).astype(np.uint8)

        # Convert to grayscale
        if frame_np.shape[2] == 4:
            gray = cv2.cvtColor(frame_uint8[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)

        # Threshold
        _, binary = cv2.threshold(gray, int(sensitivity * 255), 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create mask with largest contour
        mask = np.zeros_like(gray)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        return mask

    def _get_mask_from_alpha(self, frame_np, alpha_np, sensitivity):
        """Use provided alpha channel as mask."""
        if alpha_np is not None:
            # Alpha is already provided, use it
            mask = (alpha_np * 255).astype(np.uint8)
            # Apply sensitivity threshold
            mask = (mask > int(sensitivity * 255)).astype(np.uint8) * 255
        elif frame_np.shape[2] == 4:
            # Extract alpha from frame
            alpha = frame_np[:, :, 3]
            mask = (alpha * 255).astype(np.uint8)
            mask = (mask > int(sensitivity * 255)).astype(np.uint8) * 255
        else:
            # No alpha available, fall back to bbox detection
            mask = self._detect_bbox_from_pixels(frame_np, sensitivity)

        return mask

    def _mask_to_bbox(self, mask):
        """Convert mask to bounding box coordinates."""
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        return x, y, x + w, y + h

    def _resize_with_padding(self, img, alpha, target_size):
        """Resize image and alpha to fit within target_size while preserving aspect ratio."""
        tw, th = target_size
        h, w = img.shape[:2]

        scale = min(tw / w, th / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        if alpha.ndim == 2:
            resized_alpha = cv2.resize(
                alpha, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
            )
        else:
            resized_alpha = cv2.resize(
                alpha, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
            )

        canvas_img = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
        canvas_alpha = np.zeros((th, tw), dtype=alpha.dtype)

        x0 = (tw - new_w) // 2
        y0 = (th - new_h) // 2

        canvas_img[y0 : y0 + new_h, x0 : x0 + new_w] = resized_img
        canvas_alpha[y0 : y0 + new_h, x0 : x0 + new_w] = resized_alpha

        return canvas_img, canvas_alpha

    def auto_crop(
        self,
        frames: torch.Tensor,
        method: str,
        sensitivity: float,
        padding: int,
        alpha: torch.Tensor = None,
    ):
        """Auto-crop frames to content bounds with padding."""
        print(f"[AutoCropper] Processing {frames.shape[0]} frames using {method}...")

        # Only load model if using anime_seg method
        if method == "anime_seg":
            model, device = self._get_model()
        else:
            model, device = None, None

        frames_np = frames.cpu().numpy()

        if frames_np.ndim != 4:
            raise ValueError(
                f"Expected frames with shape (N, H, W, C), got {frames_np.shape}"
            )

        # Handle optional alpha input
        if alpha is not None:
            alpha_np = alpha.cpu().numpy()
            if alpha_np.ndim == 4 and alpha_np.shape[-1] == 1:
                alpha_np = alpha_np[..., 0]
            if alpha_np.ndim != 3:
                raise ValueError(
                    f"Expected alpha with shape (N, H, W), got {alpha_np.shape}"
                )
        else:
            # Create a full opacity alpha if not provided
            alpha_np = np.ones(
                (frames_np.shape[0], frames_np.shape[1], frames_np.shape[2]),
                dtype=np.float32,
            )

        num_frames, H, W, C = frames_np.shape

        print(f"[AutoCropper] Analyzing frames with sensitivity {sensitivity}...")
        global_box = None

        for i, frame in enumerate(frames_np):
            # Get frame-specific alpha if available
            frame_alpha = alpha_np[i] if alpha_np is not None else None

            # Generate mask based on selected method
            if method == "anime_seg":
                mask = self._segment_frame_anime(frame, model, device, sensitivity)
            elif method == "alpha_channel":
                mask = self._get_mask_from_alpha(frame, frame_alpha, sensitivity)
            elif method == "bbox_detection":
                mask = self._detect_bbox_from_pixels(frame, sensitivity)
            elif method == "contour_detection":
                mask = self._detect_contour(frame, sensitivity)
            else:
                raise ValueError(f"Unknown method: {method}")

            bbox = self._mask_to_bbox(mask)

            if bbox is None:
                continue

            if global_box is None:
                global_box = list(bbox)
            else:
                global_box[0] = min(global_box[0], bbox[0])
                global_box[1] = min(global_box[1], bbox[1])
                global_box[2] = max(global_box[2], bbox[2])
                global_box[3] = max(global_box[3], bbox[3])

            if (i + 1) % 10 == 0:
                print(f"[AutoCropper] Processed {i + 1}/{num_frames} frames")

        if global_box is None:
            raise RuntimeError("No content detected in any frame")

        x1, y1, x2, y2 = global_box

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(W, x2 + padding)
        y2 = min(H, y2 + padding)

        crop_width = x2 - x1
        crop_height = y2 - y1

        print(
            f"[AutoCropper] Crop region: ({x1}, {y1}) -> ({x2}, {y2}), size: {crop_width}x{crop_height}"
        )

        cropped_frames = []
        cropped_alphas = []

        for i in range(num_frames):
            frame = frames_np[i]
            alpha_frame = alpha_np[i]

            cropped_frame = frame[y1:y2, x1:x2]
            cropped_alpha = alpha_frame[y1:y2, x1:x2]

            cropped_frames.append(cropped_frame)
            cropped_alphas.append(cropped_alpha)

        result_frames = torch.from_numpy(np.stack(cropped_frames).astype(np.float32))
        result_alphas = torch.from_numpy(np.stack(cropped_alphas).astype(np.float32))

        final_width = result_frames.shape[2]
        final_height = result_frames.shape[1]

        print(f"[AutoCropper] Done. Output size: {final_width}x{final_height}")

        return (result_frames, result_alphas, final_width, final_height)
