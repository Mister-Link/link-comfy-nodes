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


def _registration_signal(frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    luminance = frame[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return (0.35 * alpha + 0.65 * luminance * alpha).astype(np.float32)


def _pairwise_shift(previous: np.ndarray, current: np.ndarray) -> tuple[float, float, float]:
    window = cv2.createHanningWindow((previous.shape[1], previous.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(previous, current, window)
    return float(shift[0]), float(shift[1]), float(response)


class StabilizeFramesNode:
    CATEGORY = "Image/Animation"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("frames", "masks", "stabilization_metadata")
    FUNCTION = "stabilize"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
        }}

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

        scale = min(REFERENCE_CANVAS_W / max(widths), REFERENCE_CANVAS_H / max(heights)) * FILL_MARGIN
        scaled_frames, scaled_alphas = [], []
        for frame, alpha in zip(frames, alphas):
            size = (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale)))
            scaled_frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4))
            scaled_alphas.append(cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR))

        # Register adjacent frames around the complete loop. This is global,
        # feature-independent motion estimation; no object-specific anchor.
        signals = [_registration_signal(frame, alpha) for frame, alpha in zip(scaled_frames, scaled_alphas)]
        edges = np.asarray([_pairwise_shift(a, b) for a, b in zip(signals, signals[1:] + signals[:1])], dtype=np.float32)
        edges[:, :2] /= max(scale, 1e-8)
        edges[:, :2] -= np.mean(edges[:, :2], axis=0)
        positions = np.zeros((len(frames), 2), dtype=np.float32)
        for i, edge in enumerate(edges[:-1], start=1):
            positions[i] = positions[i - 1] + edge[:2]
        positions -= np.median(positions, axis=0)
        corrections = -positions
        # The image is moved by `corrections` to remove the source motion.
        # The game later adds this inverse displacement back per frame, after
        # pixelization and spritesheet trimming.
        motion_offsets = positions

        margin = int(max(64, np.ceil(np.max(np.abs(corrections))) + 4))
        prepared = []
        for frame, alpha, (dx, dy) in zip(scaled_frames, scaled_alphas, corrections):
            h, w = frame.shape[:2]
            transform = np.float32([[1, 0, dx + margin], [0, 1, dy + margin]])
            corrected_frame = cv2.warpAffine(frame, transform, (w + 2 * margin, h + 2 * margin), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
            corrected_alpha = cv2.warpAffine(alpha, transform, (w + 2 * margin, h + 2 * margin), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            prepared.append((corrected_frame, corrected_alpha))

        union = np.max(np.stack([alpha for _, alpha in prepared]), axis=0)
        ys, xs = np.where(union > 0.02)
        if ys.size == 0:
            raise ValueError("Registration moved all frames outside their canvas")
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        result, output_masks, manifest_frames = [], [], []
        for index, ((frame, alpha), (motion_x, motion_y)) in enumerate(zip(prepared, motion_offsets)):
            cropped = frame[y1:y2, x1:x2, :3]
            cropped_alpha = alpha[y1:y2, x1:x2]
            result.append(np.concatenate([cropped, cropped_alpha[..., None]], axis=-1))
            output_masks.append(cropped_alpha)
            manifest_frames.append({
                "index": index,
                "spriteSourceSize": {"x": 0, "y": 0, "w": x2 - x1, "h": y2 - y1},
                "motionOffset": {"x": float(motion_x), "y": float(motion_y)},
            })

        metadata = {"format": "link-comfy-nodes/stabilization-v1", "sourceSize": {"w": x2 - x1, "h": y2 - y1}, "pivot": {"x": round((x2 - x1) / 2), "y": round((y2 - y1) / 2)}, "frames": manifest_frames}
        output = torch.from_numpy(np.stack(result)).to(device=image.device, dtype=image.dtype)
        output_masks = torch.from_numpy(np.stack(output_masks)).to(device=mask.device, dtype=mask.dtype)
        return output, output_masks, json.dumps(metadata)
