from __future__ import annotations

import cv2
import numpy as np
import torch

import folder_paths  # type: ignore[import-untyped]

try:
    from ...models.model_utils import load_anime_seg_model

    ANIME_SEG_AVAILABLE = True
except Exception as e:
    ANIME_SEG_AVAILABLE = False
    print(f"[AutoCropper] AnimeSegmentation not available: {e}")


class AutoCropperNode:
    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_frames", "cropped_alpha", "width", "height", "bbox_x", "bbox_y", "bbox_w", "bbox_h")
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
                        "bbox",
                        "anime",
                        "bg_sub",
                    ],
                    {"default": "bg_sub"},
                ),
                "sensitivity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "padding": (
                    "INT",
                    {"default": 5, "min": 0, "max": 500, "step": 1},
                ),
                "padding_color": (
                    "STRING",
                    {"default": "#000000"},
                ),
                "use_image_padding": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If true, padding area is taken from the original frame when in bounds; "
                            "out-of-bounds area uses padding_color."
                        ),
                    },
                ),
            },
            "optional": {
                "alpha": ("MASK",),
            },
        }

    @classmethod
    def _get_model(cls):
        if not ANIME_SEG_AVAILABLE:
            raise RuntimeError(
                "AnimeSegmentation model not available. "
                "Please ensure the models folder is properly set up."
            )

        if cls._model is None:
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model = load_anime_seg_model(folder_paths, cls._device)

        return cls._model, cls._device

    @torch.no_grad()
    def _segment_frame_anime(self, frame_np, model, device, sensitivity):
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

    def _detect_bbox(self, frame_np, sensitivity):
        frame_uint8 = (frame_np * 255).astype(np.uint8)
        if frame_np.shape[2] >= 3:
            rgb_channels = frame_uint8[:, :, :3]
        else:
            rgb_channels = frame_uint8
        mask = (
            np.any(rgb_channels > int(sensitivity * 255), axis=2).astype(np.uint8) * 255
        )
        return mask

    def _detect_background_subtraction(self, frame_np, sensitivity):
        h, w = frame_np.shape[:2]
        rgb = frame_np[:, :, :3] if frame_np.shape[2] >= 3 else frame_np

        sensitivity = float(np.clip(sensitivity, 0.0, 1.0))

        # Sample background from corners (most reliable for uniform backgrounds)
        corner_size = max(2, int(min(h, w) * 0.05))
        corners = [
            rgb[:corner_size, :corner_size],
            rgb[:corner_size, -corner_size:],
            rgb[-corner_size:, :corner_size],
            rgb[-corner_size:, -corner_size:],
        ]
        corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
        bg_color = np.median(corner_pixels, axis=0)

        # Simple threshold: pixels far enough from background are foreground
        # sensitivity: 0.0 = very permissive (small threshold), 1.0 = very strict (large threshold)
        diff = np.linalg.norm(rgb - bg_color[None, None, :], axis=2)

        # Map sensitivity to threshold range
        # Low sensitivity: threshold ~0.02 (pick up any difference)
        # High sensitivity: threshold ~0.25 (only major differences)
        base_threshold = 0.02 + (sensitivity * 0.23)
        fg_mask = (diff > base_threshold).astype(np.uint8) * 255

        # Clean up: close small holes, erode to remove noise, dilate to restore size
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_k)

        # Erode to remove thin noise
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        fg_mask = cv2.erode(fg_mask, erode_k, iterations=1)

        # Dilate to restore edges and connect small gaps
        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.dilate(fg_mask, dilate_k, iterations=2)

        return fg_mask

    def _mask_to_bbox(self, mask):
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        return x, y, x + w, y + h

    def _resize_with_padding(self, img, alpha, target_size):
        tw, th = target_size
        h, w = img.shape[:2]

        scale = min(tw / w, th / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
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

    def _hex_to_rgb(self, hex_color: str):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r, g, b = (
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16),
            )
            return (r / 255.0, g / 255.0, b / 255.0)
        return (0.0, 0.0, 0.0)

    def auto_crop(
        self,
        frames: torch.Tensor,
        method: str,
        sensitivity: float,
        padding: int,
        padding_color: str,
        use_image_padding: bool,
        alpha: torch.Tensor = None,
    ):
        print(f"[AutoCropper] Processing {frames.shape[0]} frames using {method}...")

        # Backward compatibility for older workflow values.
        if method == "anime_seg":
            method = "anime"
        elif method == "background_subtraction":
            method = "bg_sub"
        elif method in ("bbox_detection", "alpha_channel", "contour_detection"):
            method = "bbox"

        if method == "anime":
            model, device = self._get_model()
        else:
            model, device = None, None

        frames_np = frames.cpu().numpy()

        if frames_np.ndim != 4:
            raise ValueError(
                f"Expected frames with shape (N, H, W, C), got {frames_np.shape}"
            )

        if alpha is not None:
            alpha_np = alpha.cpu().numpy()
            if alpha_np.ndim == 4 and alpha_np.shape[-1] == 1:
                alpha_np = alpha_np[..., 0]
            if alpha_np.ndim != 3:
                raise ValueError(
                    f"Expected alpha with shape (N, H, W), got {alpha_np.shape}"
                )
        else:
            alpha_np = np.ones(
                (frames_np.shape[0], frames_np.shape[1], frames_np.shape[2]),
                dtype=np.float32,
            )

        num_frames, H, W, C = frames_np.shape

        print(f"[AutoCropper] Analyzing frames with sensitivity {sensitivity}...")
        global_box = None

        for i, frame in enumerate(frames_np):
            if method == "anime":
                mask = self._segment_frame_anime(frame, model, device, sensitivity)
            elif method == "bg_sub":
                mask = self._detect_background_subtraction(frame, sensitivity)
            elif method == "bbox":
                mask = self._detect_bbox(frame, sensitivity)
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

        crop_width = x2 - x1
        crop_height = y2 - y1

        print(
            f"[AutoCropper] Content bounds: ({x1}, {y1}) -> ({x2}, {y2}), "
            f"size: {crop_width}x{crop_height}"
        )

        cropped_frames = []
        cropped_alphas = []

        for i in range(num_frames):
            frame = frames_np[i]
            alpha_frame = alpha_np[i]

            cropped_frame = frame[y1:y2, x1:x2]
            cropped_alpha = alpha_frame[y1:y2, x1:x2]

            if padding > 0:
                padded_h = crop_height + (padding * 2)
                padded_w = crop_width + (padding * 2)

                r, g, b = self._hex_to_rgb(padding_color)

                padded_frame = np.zeros(
                    (padded_h, padded_w, C), dtype=cropped_frame.dtype
                )
                if C >= 3:
                    padded_frame[:, :, 0] = r
                    padded_frame[:, :, 1] = g
                    padded_frame[:, :, 2] = b
                if C == 4:
                    padded_frame[:, :, 3] = 0.0

                padded_alpha = np.zeros((padded_h, padded_w), dtype=cropped_alpha.dtype)

                if use_image_padding:
                    src_x1 = x1 - padding
                    src_y1 = y1 - padding
                    src_x2 = x2 + padding
                    src_y2 = y2 + padding

                    # Copy only the in-bounds part from the original frame.
                    ix1 = max(0, src_x1)
                    iy1 = max(0, src_y1)
                    ix2 = min(W, src_x2)
                    iy2 = min(H, src_y2)

                    if ix2 > ix1 and iy2 > iy1:
                        dx1 = ix1 - src_x1
                        dy1 = iy1 - src_y1
                        dx2 = dx1 + (ix2 - ix1)
                        dy2 = dy1 + (iy2 - iy1)
                        padded_frame[dy1:dy2, dx1:dx2] = frame[iy1:iy2, ix1:ix2]
                        padded_alpha[dy1:dy2, dx1:dx2] = alpha_frame[iy1:iy2, ix1:ix2]
                else:
                    padded_frame[
                        padding : padding + crop_height, padding : padding + crop_width
                    ] = cropped_frame
                    padded_alpha[
                        padding : padding + crop_height, padding : padding + crop_width
                    ] = cropped_alpha

                cropped_frames.append(padded_frame)
                cropped_alphas.append(padded_alpha)
            else:
                cropped_frames.append(cropped_frame)
                cropped_alphas.append(cropped_alpha)

        result_frames = torch.from_numpy(np.stack(cropped_frames).astype(np.float32))
        result_alphas = torch.from_numpy(np.stack(cropped_alphas).astype(np.float32))

        final_width = result_frames.shape[2]
        final_height = result_frames.shape[1]

        print(
            f"[AutoCropper] Done. Output size: {final_width}x{final_height} "
            f"(with {padding}px padding)"
        )

        return (result_frames, result_alphas, final_width, final_height, x1, y1, crop_width, crop_height)
