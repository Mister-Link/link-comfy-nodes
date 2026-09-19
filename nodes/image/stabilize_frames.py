from __future__ import annotations

import json

import cv2
import numpy as np
import torch

# Border of transparent pixels kept between the stabilized content and the
# canvas edge. A few pixels are needed because Lanczos resampling has a small
# ringing footprint beyond the detected foreground bounds.
PADDING_PX = 4

# Below this cv2.phaseCorrelate response, a pairwise shift is treated as
# unreliable (occlusion, motion blur, low-texture frame pair) and dropped to
# zero rather than trusted. Because positions are a cumulative sum of edges,
# one bad edge otherwise poisons every subsequent frame's position, not just
# the pair it was measured from.
RESPONSE_THRESHOLD = 0.10

# Moving-average window applied to the measured position curve. Even a
# "good" phase-correlate match has sub-pixel measurement noise; without
# damping that noise is applied to the output at full strength every frame,
# which is what shows up as a few pixels of jitter. Five frames suppresses
# isolated registration noise while preserving the broad displacement of a
# jump or other one-shot action.
SMOOTHING_WINDOW = 5


def _foreground(mask: np.ndarray):
    mask_u8 = np.clip(mask * 255, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(mask_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
    bg_white = np.median(border) > 127
    foreground = (binary == 0 if bg_white else binary == 255).astype(np.uint8)
    source_alpha = 1.0 - mask if bg_white else mask
    visible = source_alpha > (1.0 / 255.0)
    ys, xs = np.where(visible)
    if len(xs):
        full_bounds = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    else:
        h, w = mask.shape
        full_bounds = (0, 0, w, h)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if count <= 1:
        return source_alpha, source_alpha, full_bounds
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == label).astype(np.uint8)
    keep = cv2.dilate(component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1).astype(np.float32)
    alpha = source_alpha * keep
    # Use the complete alpha bounds for canvas sizing even though the largest
    # component is used for registration. A disconnected hat, hand, prop, or
    # effect must never be cropped just because it is not the largest island.
    return alpha, source_alpha, full_bounds


def _registration_signal(frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    luminance = frame[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    signal = 0.35 * alpha + 0.65 * luminance * alpha
    # Blur out per-pixel noise (antialiasing, compression, per-frame Otsu
    # threshold flicker on the mask edge) that phaseCorrelate would otherwise
    # register as motion. Matches the intent of the old distance-transform +
    # GaussianBlur(sigma=4) signal this replaced, just applied to the new
    # luminance-weighted signal instead of a binary mask.
    signal = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 2.0)

    return signal


def _pairwise_shift(previous: np.ndarray, current: np.ndarray) -> tuple[float, float, float]:
    window = cv2.createHanningWindow((previous.shape[1], previous.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(previous, current, window)
    return float(shift[0]), float(shift[1]), float(response)


def _smooth_linear(positions: np.ndarray, window: int) -> np.ndarray:
    """Moving average for a one-shot sequence without wrapping endpoints."""
    n = len(positions)
    if n <= window:
        return positions
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.empty_like(positions)
    for axis in range(positions.shape[1]):
        padded = np.pad(positions[:, axis], (pad, pad), mode="edge")
        smoothed[:, axis] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _registration_positions(
    scaled_frames: list[np.ndarray],
    scaled_alphas: list[np.ndarray],
) -> np.ndarray:
    """Measure open-chain global positions in the scaled frame space.

    Playback behavior is deliberately not part of stabilization metadata. The
    registration pass therefore treats the batch as an ordered sequence and
    never invents a closing edge between the last and first frame.
    """
    signals = [
        _registration_signal(frame, alpha)
        for frame, alpha in zip(scaled_frames, scaled_alphas)
    ]
    if len(signals) <= 1:
        return np.zeros((len(signals), 2), dtype=np.float32)

    edges = np.asarray(
        [_pairwise_shift(a, b) for a, b in zip(signals, signals[1:])],
        dtype=np.float32,
    )
    edges[edges[:, 2] < RESPONSE_THRESHOLD, :2] = 0.0
    positions = np.zeros((len(signals), 2), dtype=np.float32)
    for index, edge in enumerate(edges, start=1):
        positions[index] = positions[index - 1] + edge[:2]
    positions -= np.median(positions, axis=0)
    return _smooth_linear(positions, SMOOTHING_WINDOW)


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
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "anchor": ("STRING", {"default": ""}),
            },
        }

    @staticmethod
    def _parse_anchor(anchor: str, width: int, height: int):
        if not anchor:
            return None
        try:
            payload = json.loads(anchor) if isinstance(anchor, str) else anchor
            point = payload.get("anchor", payload)
            source_size = payload.get("sourceSize", {})
            source_w = max(1.0, float(source_size.get("w", width)))
            source_h = max(1.0, float(source_size.get("h", height)))
            return np.asarray(
                [
                    float(point["x"]) * width / source_w,
                    float(point["y"]) * height / source_h,
                ],
                dtype=np.float32,
            )
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def stabilize(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        anchor: str = "",
    ):
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
        static_anchor = self._parse_anchor(anchor, w, h)

        alphas, render_alphas, boxes, widths, heights = [], [], [], [], []
        for current_mask in masks:
            alpha, render_alpha, (x1, y1, x2, y2) = _foreground(current_mask)
            alphas.append(alpha)
            render_alphas.append(render_alpha)
            boxes.append((x1, y1, x2, y2))
            widths.append(max(1, x2 - x1))
            heights.append(max(1, y2 - y1))

        # Scale content to fill the canvas (minus a small transparent
        # border) as much as possible while preserving aspect ratio - the
        # limiting axis fills to the border exactly; the other axis is
        # letterboxed with transparent padding split evenly on both sides
        # rather than stretched to match.
        if static_anchor is not None:
            # Anchored mode is intentionally deterministic. The selected
            # coordinate is a canvas contract, not a visual-tracking hint:
            # every source frame receives the same transform.
            scale = 1.0
            scaled_frames = list(frames)
            scaled_alphas = list(alphas)
            scaled_render_alphas = list(render_alphas)
            positions = np.zeros((len(frames), 2), dtype=np.float32)
        else:
            scale = min(
                (w - 2 * PADDING_PX) / max(widths),
                (h - 2 * PADDING_PX) / max(heights),
            )

            # Unanchored mode retains automatic registration.
            scaled_frames, scaled_alphas, scaled_render_alphas = [], [], []
            for frame, alpha, render_alpha in zip(frames, alphas, render_alphas):
                size = (
                    max(1, round(frame.shape[1] * scale)),
                    max(1, round(frame.shape[0] * scale)),
                )
                scaled_frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4))
                scaled_alphas.append(cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR))
                scaled_render_alphas.append(cv2.resize(render_alpha, size, interpolation=cv2.INTER_LINEAR))

            positions = _registration_positions(
                scaled_frames,
                scaled_alphas,
            )
        corrections = -positions

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

        # The anchor node supplies one fixed source/output coordinate, not a
        # per-frame point and not a replacement canvas center. Anchored mode
        # uses the identity transform here; the same source coordinate is
        # therefore in the same place for every output frame.
        output_pivot = np.asarray([w / 2.0, h / 2.0], dtype=np.float32)
        if static_anchor is not None:
            center_x = float(static_anchor[0] - static_anchor[0] * scale - corrections[0][0])
            center_y = float(static_anchor[1] - static_anchor[1] * scale - corrections[0][1])
            # The clicked point is the stabilization reference only. It is
            # commonly the feet, but the engine's render pivot is the
            # physics-body center. That center is the middle of the
            # canonical output canvas, not the clicked anchor.
            output_pivot = None

        # Compute a tight canvas from every transformed content bound. Keep
        # only a small safety border for resampling; crop transparent source
        # margins, but grow the canvas whenever anchoring needs more room.
        placed_min_x = min_x + center_x
        placed_min_y = min_y + center_y
        placed_max_x = max_x + center_x
        placed_max_y = max_y + center_y
        canvas_min_x = int(np.floor(placed_min_x)) - PADDING_PX
        canvas_min_y = int(np.floor(placed_min_y)) - PADDING_PX
        canvas_max_x = int(np.ceil(placed_max_x)) + PADDING_PX
        canvas_max_y = int(np.ceil(placed_max_y)) + PADDING_PX
        output_width = max(1, canvas_max_x - canvas_min_x)
        output_height = max(1, canvas_max_y - canvas_min_y)
        canvas_shift_x = -canvas_min_x
        canvas_shift_y = -canvas_min_y

        # Expanding the canvas adds only transparent padding. No source pixel
        # is discarded when anchor placement moves content past an old edge.
        result, output_masks, manifest_frames = [], [], []
        for index, (frame, render_alpha, (dx, dy)) in enumerate(
            zip(scaled_frames, scaled_render_alphas, corrections)
        ):
            transform = np.float32(
                [[1, 0, center_x + dx + canvas_shift_x],
                 [0, 1, center_y + dy + canvas_shift_y]]
            )
            corrected_frame = cv2.warpAffine(
                frame,
                transform,
                (output_width, output_height),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
            )
            corrected_alpha = cv2.warpAffine(
                render_alpha,
                transform,
                (output_width, output_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            result.append(np.concatenate([corrected_frame[:, :, :3], corrected_alpha[..., None]], axis=-1))
            output_masks.append(corrected_alpha)
            manifest_frames.append({
                "index": index,
                "spriteSourceSize": {"x": 0, "y": 0, "w": output_width, "h": output_height},
                "motionOffset": {"x": float(motion_offsets[index][0]), "y": float(motion_offsets[index][1])},
            })

        if not np.any(np.stack(output_masks) > 0.02):
            raise ValueError("Registration moved all frames outside their canvas")

        if static_anchor is not None:
            # Anchored output is normalized to the canonical frame center so
            # feet/head anchors do not become the game's physics origin.
            output_anchor = static_anchor + np.asarray([canvas_shift_x, canvas_shift_y], dtype=np.float32)
            output_pivot = np.asarray([output_width / 2.0, output_height / 2.0], dtype=np.float32)
        else:
            output_anchor = None
            output_pivot = output_pivot + np.asarray([canvas_shift_x, canvas_shift_y], dtype=np.float32)

        metadata = {
            "format": "link-comfy-nodes/stabilization-v1",
            # The stabilized sheet is anchored at the player's physics-body
            # center. Registration history remains per-frame diagnostic data;
            # the game must not reapply it as root motion by default.
            "root": {
                "type": "physics_body_center",
                "x": round(float(output_pivot[0])),
                "y": round(float(output_pivot[1])),
            },
            "rootMotion": False,
            "sourceSize": {"w": output_width, "h": output_height},
            "pivot": {
                "x": round(float(output_pivot[0])),
                "y": round(float(output_pivot[1])),
            },
            "frames": manifest_frames,
        }
        if static_anchor is not None:
            metadata["anchor"] = {
                "x": round(float(output_anchor[0])),
                "y": round(float(output_anchor[1])),
            }

        output = torch.from_numpy(np.stack(result)).to(device=image.device, dtype=image.dtype)
        output_masks = torch.from_numpy(np.stack(output_masks)).to(device=mask.device, dtype=mask.dtype)
        return output, output_masks, json.dumps(metadata)
