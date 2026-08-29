from __future__ import annotations

import json

import cv2
import numpy as np
import torch

REFERENCE_CANVAS_W = 512
REFERENCE_CANVAS_H = 1152
FILL_MARGIN = 0.98


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
    keep = cv2.dilate(
        component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1
    ).astype(np.float32)
    alpha = (1.0 - mask if bg_white else mask) * keep
    x, y, w, h, _ = stats[label]
    return alpha, (x, y, x + w, y + h)


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

        alphas, widths, heights = [], [], []
        for current_mask in masks:
            alpha, (x1, y1, x2, y2) = _foreground(current_mask)
            alphas.append(alpha)
            widths.append(max(1, x2 - x1))
            heights.append(max(1, y2 - y1))

        # One scale for the entire sequence. Per-frame scaling changes a
        # character's apparent size and destroys the original padding/motion
        # that the metadata needs to replay.
        largest_width = max(widths)
        largest_height = max(heights)
        scale = min(
            REFERENCE_CANVAS_W / largest_width,
            REFERENCE_CANVAS_H / largest_height,
        ) * FILL_MARGIN

        prepared = []
        max_width = max_height = 0
        for frame, alpha in zip(frames, alphas):
            scaled_w = max(1, round(frame.shape[1] * scale))
            scaled_h = max(1, round(frame.shape[0] * scale))
            scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
            scaled_alpha = cv2.resize(alpha, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            ys, xs = np.where(scaled_alpha > 0.02)
            if ys.size == 0:
                raise ValueError("Mask contains no foreground after cleanup")
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            content = scaled_frame[y1:y2, x1:x2, :3]
            content_alpha = scaled_alpha[y1:y2, x1:x2]
            h, w = content_alpha.shape

            # The bbox center is the sequence-wide stabilization anchor. The
            # raw center records the movement removed from the source frame.
            source_center_x = x1 + w / 2.0
            source_center_y = y1 + h / 2.0
            prepared.append((content, content_alpha, source_center_x, source_center_y))
            max_width = max(max_width, w)
            max_height = max(max_height, h)

        pivot_x = int(np.ceil(max_width / 2.0))
        pivot_y = int(np.ceil(max_height / 2.0))
        reference_center_x = float(np.median([entry[2] for entry in prepared]))
        reference_center_y = float(np.median([entry[3] for entry in prepared]))

        placements = []
        output_w = output_h = 0
        for content, content_alpha, source_x, source_y in prepared:
            h, w = content_alpha.shape
            dst_x = round(pivot_x - w / 2.0)
            dst_y = round(pivot_y - h / 2.0)
            placements.append((
                content,
                content_alpha,
                dst_x,
                dst_y,
                source_x - reference_center_x,
                source_y - reference_center_y,
            ))
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

