from __future__ import annotations

import json

import cv2
import numpy as np
import torch

# Border of transparent pixels kept between the stabilized content and the
# canvas edge. A few pixels are needed because Lanczos resampling has a small
# ringing footprint beyond the detected foreground bounds.
PADDING_PX = 4


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


def _registration_positions(
    scaled_frames: list[np.ndarray],
    scaled_alphas: list[np.ndarray],
) -> np.ndarray:
    """Measure global loop positions in the same space as the scaled frames."""
    signals = [
        _registration_signal(frame, alpha)
        for frame, alpha in zip(scaled_frames, scaled_alphas)
    ]
    if len(signals) <= 1:
        return np.zeros((len(signals), 2), dtype=np.float32)

    edges = np.asarray(
        [_pairwise_shift(a, b) for a, b in zip(signals, signals[1:] + signals[:1])],
        dtype=np.float32,
    )
    edges[:, :2] -= np.mean(edges[:, :2], axis=0)
    positions = np.zeros((len(signals), 2), dtype=np.float32)
    for index, edge in enumerate(edges[:-1], start=1):
        positions[index] = positions[index - 1] + edge[:2]
    positions -= np.median(positions, axis=0)
    return positions


def _corrected_content_bounds(
    boxes: list[tuple[int, int, int, int]],
    scale: float,
    corrections: np.ndarray,
) -> tuple[float, float, float, float]:
    """Find the union of all foreground boxes after stabilization motion."""
    bounds = np.asarray(
        [
            (
                x1 * scale + dx,
                y1 * scale + dy,
                x2 * scale + dx,
                y2 * scale + dy,
            )
            for (x1, y1, x2, y2), (dx, dy) in zip(boxes, corrections)
        ],
        dtype=np.float32,
    )
    return (
        float(bounds[:, 0].min()),
        float(bounds[:, 1].min()),
        float(bounds[:, 2].max()),
        float(bounds[:, 3].max()),
    )


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

        # The canvas the caller handed us is also the canvas we hand back -
        # stabilization must only ever translate content to cancel jitter,
        # never resize or crop it. Two separate generation batches that get
        # normalized against a shared fixed reference canvas (the old
        # behavior) end up at different effective zoom levels depending on
        # each batch's own detected foreground size, which is what caused
        # idle_r_alt2 to render visibly wider than idle_r/idle_r_alt despite
        # a nominally identical frame size.
        h, w = frames.shape[1:3]

        alphas, boxes, widths, heights = [], [], [], []
        for current_mask in masks:
            alpha, (x1, y1, x2, y2) = _foreground(current_mask)
            alphas.append(alpha)
            boxes.append((x1, y1, x2, y2))
            widths.append(max(1, x2 - x1))
            heights.append(max(1, y2 - y1))

        # Scale content to fill the canvas (minus a small transparent
        # border) as much as possible while preserving aspect ratio - the
        # limiting axis fills to the border exactly; the other axis is
        # letterboxed with transparent padding split evenly on both sides
        # rather than stretched to match.
        scale = min(
            (w - 2 * PADDING_PX) / max(widths),
            (h - 2 * PADDING_PX) / max(heights),
        )

        # Registration corrections can make the union wider or taller than
        # the largest individual foreground box. Recompute at a fitted scale
        # so the corrected union is guaranteed to fit before placement.
        for _ in range(4):
            scaled_frames, scaled_alphas = [], []
            for frame, alpha in zip(frames, alphas):
                size = (
                    max(1, round(frame.shape[1] * scale)),
                    max(1, round(frame.shape[0] * scale)),
                )
                scaled_frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4))
                scaled_alphas.append(cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR))

            positions = _registration_positions(scaled_frames, scaled_alphas)
            corrections = -positions
            min_x, min_y, max_x, max_y = _corrected_content_bounds(
                boxes,
                scale,
                corrections,
            )
            content_width = max(1.0, max_x - min_x)
            content_height = max(1.0, max_y - min_y)
            fit_scale = min(
                (w - 2 * PADDING_PX) / content_width,
                (h - 2 * PADDING_PX) / content_height,
                1.0,
            )
            if fit_scale >= 0.999:
                break
            scale *= fit_scale * 0.999

        # The image is moved by `corrections` to remove the source motion.
        # The game later adds this inverse displacement back per frame, after
        # pixelization and spritesheet trimming. Reported in the same units
        # as sourceSize (original canvas pixels), not the internal scaled
        # working space.
        motion_offsets = positions / scale

        # Center the corrected foreground union, rather than the full source
        # canvas. This removes arbitrary source padding (especially above the
        # subject) while keeping every corrected pose inside the output.
        min_x, min_y, max_x, max_y = _corrected_content_bounds(
            boxes,
            scale,
            corrections,
        )
        center_x = PADDING_PX + (w - 2 * PADDING_PX - (max_x - min_x)) / 2.0 - min_x
        center_y = PADDING_PX + (h - 2 * PADDING_PX - (max_y - min_y)) / 2.0 - min_y

        result, output_masks, manifest_frames = [], [], []
        for index, (frame, alpha, (dx, dy)) in enumerate(zip(scaled_frames, scaled_alphas, corrections)):
            transform = np.float32([[1, 0, center_x + dx], [0, 1, center_y + dy]])
            corrected_frame = cv2.warpAffine(frame, transform, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
            corrected_alpha = cv2.warpAffine(alpha, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            result.append(np.concatenate([corrected_frame[:, :, :3], corrected_alpha[..., None]], axis=-1))
            output_masks.append(corrected_alpha)
            manifest_frames.append({
                "index": index,
                "spriteSourceSize": {"x": 0, "y": 0, "w": w, "h": h},
                "motionOffset": {"x": float(motion_offsets[index][0]), "y": float(motion_offsets[index][1])},
            })

        if not np.any(np.stack(output_masks) > 0.02):
            raise ValueError("Registration moved all frames outside their canvas")

        metadata = {
            "format": "link-comfy-nodes/stabilization-v1",
            "sourceSize": {"w": w, "h": h},
            "pivot": {"x": round(w / 2), "y": round(h / 2)},
            "frames": manifest_frames,
        }
        output = torch.from_numpy(np.stack(result)).to(device=image.device, dtype=image.dtype)
        output_masks = torch.from_numpy(np.stack(output_masks)).to(device=mask.device, dtype=mask.dtype)
        return output, output_masks, json.dumps(metadata)
