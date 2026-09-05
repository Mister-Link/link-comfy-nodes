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
    keep = cv2.dilate(component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1).astype(np.float32)
    alpha = (1.0 - mask if bg_white else mask) * keep
    x, y, w, h, _ = stats[label]
    return alpha, (x, y, x + w, y + h)


def _registration_signal(alpha: np.ndarray) -> np.ndarray:
    binary = (alpha > 0.20).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return cv2.GaussianBlur(distance, (0, 0), 4).astype(np.float32)


def _correction_to_reference(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        score, warp = cv2.findTransformECC(reference, current, warp, cv2.MOTION_TRANSLATION, criteria)
        if not np.isfinite(score) or score < 0.50:
            return 0.0, 0.0
        return -float(warp[0, 2]), -float(warp[1, 2])
    except cv2.error:
        return 0.0, 0.0


class StabilizeFramesNode:
    CATEGORY = "Image/Animation"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("frames", "masks", "stabilization_metadata")
    FUNCTION = "stabilize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "stabilization_strength": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    def stabilize(self, image: torch.Tensor, mask: torch.Tensor, stabilization_strength: float = 0.90):
        frames = image.detach().cpu().numpy().astype(np.float32)
        masks = mask.detach().cpu().numpy().astype(np.float32)
        strength = float(np.clip(stabilization_strength, 0.0, 1.0))
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

        scale = min(REFERENCE_CANVAS_W / max(widths), REFERENCE_CANVAS_H / max(heights)) * FILL_MARGIN
        scaled_frames, scaled_alphas = [], []
        for frame, alpha in zip(frames, alphas):
            scaled_w = max(1, round(frame.shape[1] * scale))
            scaled_h = max(1, round(frame.shape[0] * scale))
            scaled_frames.append(cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4))
            scaled_alphas.append(cv2.resize(alpha, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR))

        signals = [_registration_signal(alpha) for alpha in scaled_alphas]
        reference = signals[len(signals) // 2]
        corrections = [_correction_to_reference(reference, signal) for signal in signals]

        raw_motion_x = np.asarray([-x for x, _ in corrections], dtype=np.float32)
        raw_motion_y = np.asarray([-y for _, y in corrections], dtype=np.float32)
        raw_motion_x -= np.median(raw_motion_x)
        raw_motion_y -= np.median(raw_motion_y)
        residual_scale = 1.0 - strength

        prepared = []
        for frame, alpha, (correction_x, correction_y), motion_x, motion_y in zip(scaled_frames, scaled_alphas, corrections, raw_motion_x, raw_motion_y):
            transform = np.array([[1.0, 0.0, correction_x], [0.0, 1.0, correction_y]], dtype=np.float32)
            corrected_frame = cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
            corrected_alpha = cv2.warpAffine(alpha, transform, (alpha.shape[1], alpha.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            ys, xs = np.where(corrected_alpha > 0.02)
            if ys.size == 0:
                raise ValueError("Registration moved a frame completely outside its canvas")
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            prepared.append((corrected_frame[y1:y2, x1:x2, :3], corrected_alpha[y1:y2, x1:x2], float(motion_x * residual_scale), float(motion_y * residual_scale)))

        max_width = max(alpha.shape[1] for _, alpha, _, _ in prepared)
        max_height = max(alpha.shape[0] for _, alpha, _, _ in prepared)
        pivot_x = int(np.ceil(max_width / 2.0))
        pivot_y = int(np.ceil(max_height / 2.0))
        placements, output_w, output_h = [], 0, 0
        for content, content_alpha, motion_x, motion_y in prepared:
            h, w = content_alpha.shape
            dst_x, dst_y = round(pivot_x - w / 2.0), round(pivot_y - h / 2.0)
            placements.append((content, content_alpha, dst_x, dst_y, motion_x, motion_y))
            output_w, output_h = max(output_w, dst_x + w), max(output_h, dst_y + h)

        result, output_masks, manifest_frames = [], [], []
        for index, (content, content_alpha, dst_x, dst_y, motion_x, motion_y) in enumerate(placements):
            h, w = content_alpha.shape
            canvas = np.zeros((output_h, output_w, 4), dtype=np.float32)
            canvas[dst_y:dst_y + h, dst_x:dst_x + w, :3] = content
            canvas[dst_y:dst_y + h, dst_x:dst_x + w, 3] = content_alpha
            result.append(canvas)
            output_masks.append(canvas[..., 3])
            manifest_frames.append({"index": index, "spriteSourceSize": {"x": dst_x, "y": dst_y, "w": w, "h": h}, "motionOffset": {"x": motion_x, "y": motion_y}})

        metadata = {
            "format": "link-comfy-nodes/stabilization-v1",
            "sourceSize": {"w": output_w, "h": output_h},
            "pivot": {"x": round(output_w / 2), "y": round(output_h / 2)},
            "stabilizationStrength": strength,
            "frames": manifest_frames,
        }
        return (torch.from_numpy(np.stack(result)).to(device=image.device, dtype=image.dtype), torch.from_numpy(np.stack(output_masks)).to(device=mask.device, dtype=mask.dtype), json.dumps(metadata))

