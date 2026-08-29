from __future__ import annotations

import json

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
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("frames", "masks", "stabilization_metadata")
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
                raise ValueError(f"Mask for frame {i} contains no foreground after cleanup")
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            content = scaled_frame[y1:y2, x1:x2, :3]
            content_alpha = scaled_alpha[y1:y2, x1:x2]
            local_anchor_x, local_anchor_y = _core_anchor(content_alpha)
            h, w = content_alpha.shape
            # This is the original high-resolution core position before the
            # frame is re-anchored. Its sequence-relative movement is emitted
            # as metadata so runtime playback can restore intentional bobbing.
            source_anchor_x = cx1 + local_anchor_x
            source_anchor_y = cy1 + local_anchor_y
            prepared.append((content, content_alpha, local_anchor_x, local_anchor_y, source_anchor_x, source_anchor_y))
            max_left = max(max_left, local_anchor_x)
            max_right = max(max_right, w - local_anchor_x)
            max_top = max(max_top, local_anchor_y)
            max_bottom = max(max_bottom, h - local_anchor_y)

        # Use the exact integer union of the rounded placements. Rounding the
        # floating-point core anchor independently can otherwise put an edge
        # frame at -1 or one pixel beyond a no-padding canvas.
        pivot_x = int(np.ceil(max_left))
        pivot_y = int(np.ceil(max_top))
        reference_anchor_x = float(np.median([entry[4] for entry in prepared]))
        reference_anchor_y = float(np.median([entry[5] for entry in prepared]))
        placements = []
        output_w = output_h = 0
        for content, content_alpha, local_x, local_y, source_x, source_y in prepared:
            h, w = content_alpha.shape
            dst_x = round(pivot_x - local_x)
            dst_y = round(pivot_y - local_y)
            motion_x = round(source_x - reference_anchor_x)
            motion_y = round(source_y - reference_anchor_y)
            placements.append((content, content_alpha, dst_x, dst_y, motion_x, motion_y))
            output_w = max(output_w, dst_x + w)
            output_h = max(output_h, dst_y + h)

        result, output_masks, manifest_frames = [], [], []
        for index, (content, content_alpha, dst_x, dst_y, motion_x, motion_y) in enumerate(placements):
            h, w = content_alpha.shape
            canvas = np.zeros((output_h, output_w, 4), dtype=np.float32)
            canvas[dst_y:dst_y + h, dst_x:dst_x + w, :3] = content
            canvas[dst_y:dst_y + h, dst_x:dst_x + w, 3] = content_alpha
            result.append(canvas)
            output_masks.append(canvas[..., 3])
            manifest_frames.append({
                "index": index,
                "spriteSourceSize": {"x": dst_x, "y": dst_y, "w": w, "h": h},
                "motionOffset": {"x": motion_x, "y": motion_y},
            })

        # `pivot_x/y` is the internal thick-core anchor used to remove
        # generation drift. Runtime entities in the game are historically
        # positioned at the center of their frame, so expose that independent
        # render pivot while retaining the core anchor for diagnostics.
        metadata = {
            "format": "link-comfy-nodes/stabilization-v1",
            "sourceSize": {"w": output_w, "h": output_h},
            "pivot": {"x": round(output_w / 2), "y": round(output_h / 2)},
            "stabilizationPivot": {"x": pivot_x, "y": pivot_y},
            "frames": manifest_frames,
        }
        return (
            torch.from_numpy(np.stack(result)).to(device=image.device, dtype=image.dtype),
            torch.from_numpy(np.stack(output_masks)).to(device=mask.device, dtype=mask.dtype),
            json.dumps(metadata),
        )

