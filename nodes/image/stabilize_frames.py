from __future__ import annotations

import cv2
import numpy as np
import torch


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    pad = window // 2
    return np.convolve(np.pad(values, pad, mode="edge"), np.ones(window) / window, mode="valid")


def _foreground(mask: np.ndarray):
    mask_u8 = np.clip(mask * 255, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(mask_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
    bg_white = np.median(border) > 127
    foreground = (binary == 0 if bg_white else binary == 255).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if count <= 1:
        h, w = mask.shape
        return mask, (0, 0, w, h)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == label).astype(np.uint8)
    keep = cv2.dilate(component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1).astype(np.float32)
    alpha = (1.0 - mask if bg_white else mask) * keep
    x, y, w, h, _ = stats[label]
    return alpha, (x, y, x + w, y + h)


def _core_anchor(alpha: np.ndarray):
    binary = (alpha > 0.20).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    weights = distance * distance
    total = float(weights.sum())
    if total <= 1e-6:
        ys, xs = np.where(alpha > 0.02)
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.indices(alpha.shape, dtype=np.float32)
    return float((xs * weights).sum() / total), float((ys * weights).sum() / total)


class StabilizeFramesNode:
    CATEGORY = "Image/Animation"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("frames", "masks")
    FUNCTION = "stabilize"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",)}}

    def stabilize(self, image: torch.Tensor, mask: torch.Tensor):
        frames = image.detach().cpu().numpy().astype(np.float32)
        masks = mask.detach().cpu().numpy().astype(np.float32)
        if masks.ndim == 4:
            masks = masks[..., 0]
        if masks.shape[0] == 1 and frames.shape[0] > 1:
            masks = np.repeat(masks, frames.shape[0], axis=0)
        if frames.shape[0] != masks.shape[0]:
            raise ValueError(f"Frame count mismatch: image={frames.shape[0]}, mask={masks.shape[0]}")
        if frames.shape[1:3] != masks.shape[1:3]:
            raise ValueError(f"Frame size mismatch: image={frames.shape[1:3]}, mask={masks.shape[1:3]}")

        alphas, heights, widths = [], [], []
        for current_mask in masks:
            alpha, (x1, y1, x2, y2) = _foreground(current_mask)
            alphas.append(alpha)
            widths.append(max(1, x2 - x1))
            heights.append(max(1, y2 - y1))

        smooth_height = _smooth(np.asarray(heights))
        smooth_width = _smooth(np.asarray(widths))
        target_height = 1152.0
        for width, height in zip(smooth_width, smooth_height):
            target_height = min(target_height, 512.0 / (width / height))
        target_height *= 0.98

        prepared = []
        max_left = max_right = max_top = max_bottom = 0.0
        for i, (frame, alpha) in enumerate(zip(frames, alphas)):
            scale = target_height / smooth_height[i]
            scaled_w = max(1, round(frame.shape[1] * scale))
            scaled_h = max(1, round(frame.shape[0] * scale))
            scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            scaled_alpha = cv2.resize(alpha, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            ys, xs = np.where(scaled_alpha > 0.02)
            if ys.size == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            content = scaled_frame[y1:y2, x1:x2, :3]
            content_alpha = scaled_alpha[y1:y2, x1:x2]
            anchor_x, anchor_y = _core_anchor(content_alpha)
            h, w = content_alpha.shape
            prepared.append((content, content_alpha, anchor_x, anchor_y))
            max_left = max(max_left, anchor_x)
            max_right = max(max_right, w - anchor_x)
            max_top = max(max_top, anchor_y)
            max_bottom = max(max_bottom, h - anchor_y)

        margin_x = round((max_left + max_right) * 0.05)
        margin_y = round((max_top + max_bottom) * 0.05)
        output_w = round(max_left + max_right + 2 * margin_x)
        output_h = round(max_top + max_bottom + 2 * margin_y)
        anchor_x = round(max_left + margin_x)
        anchor_y = round(max_top + margin_y)

        result = []
        output_masks = []
        for content, content_alpha, local_x, local_y in prepared:
            h, w = content_alpha.shape
            canvas = np.zeros((output_h, output_w, 4), dtype=np.float32)
            dst_x = round(anchor_x - local_x)
            dst_y = round(anchor_y - local_y)
            canvas[dst_y:dst_y + h, dst_x:dst_x + w, :3] = content
            canvas[dst_y:dst_y + h, dst_x:dst_x + w, 3] = content_alpha
            result.append(canvas)
            output_masks.append(canvas[..., 3])

        return (
            torch.from_numpy(np.stack(result)).to(device=image.device, dtype=image.dtype),
            torch.from_numpy(np.stack(output_masks)).to(device=mask.device, dtype=mask.dtype),
        )

