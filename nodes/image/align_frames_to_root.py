"""Align a frame batch to the scale and placement of a root frame."""

from __future__ import annotations

import cv2
import numpy as np
import torch

ALIGN_PADDING_PX = 8
ALIGN_REFINEMENT_POINTS = 20000


def _as_image_batch(frames: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(frames, torch.Tensor):
        raise ValueError(f"{name} must be an IMAGE tensor.")

    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4:
        raise ValueError(
            f"{name} must have shape (frames, height, width, channels); "
            f"received {tuple(frames.shape)}."
        )
    if frames.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one frame.")
    if frames.shape[-1] < 3:
        raise ValueError(f"{name} must have at least three image channels.")
    return frames


def _corner_background(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    corner_size = max(2, int(min(height, width) * 0.05))
    corners = np.concatenate(
        (
            frame[:corner_size, :corner_size, :3].reshape(-1, 3),
            frame[:corner_size, -corner_size:, :3].reshape(-1, 3),
            frame[-corner_size:, :corner_size, :3].reshape(-1, 3),
            frame[-corner_size:, -corner_size:, :3].reshape(-1, 3),
        ),
        axis=0,
    )
    return np.median(corners, axis=0).astype(np.float32)


def _foreground_mask(frame: np.ndarray) -> np.ndarray:
    """Return the largest connected subject region in an image."""
    if frame.shape[-1] > 3:
        alpha = frame[..., 3].astype(np.float32, copy=False)
        alpha_range = float(np.percentile(alpha, 95) - np.percentile(alpha, 5))
        if alpha_range > 0.05 and float(np.percentile(alpha, 5)) < 0.95:
            foreground = (alpha > 0.05).astype(np.uint8)
        else:
            foreground = None
    else:
        foreground = None

    if foreground is None:
        rgb = frame[..., :3].astype(np.float32, copy=False)
        background = _corner_background(frame)
        difference = np.linalg.norm(rgb - background[None, None, :], axis=2)
        foreground = (difference > 0.08).astype(np.uint8)

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if count <= 1:
        raise ValueError("Could not detect a foreground subject in the image.")

    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[label, cv2.CC_STAT_AREA])
    if area < max(16, int(frame.shape[0] * frame.shape[1] * 0.001)):
        raise ValueError("Could not detect a sufficiently large foreground subject.")
    return (labels == label).astype(np.uint8)


def _foreground_bbox(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Find the main subject against the image's corner background."""
    foreground = _foreground_mask(frame)
    ys, xs = np.where(foreground)
    if len(xs) == 0:
        raise ValueError("Could not detect a foreground subject in the image.")
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )


def _alignment_transform(
    root_frame: np.ndarray,
    reference_frame: np.ndarray,
    match_axis: str,
) -> np.ndarray:
    """Build a fixed scale/anchor transform from root and the first frame.

    The subject, rather than the source canvas, determines the scale. The
    first frame is independently scaled on X and Y to match the root
    silhouette bounds, then anchored by the selected axis. That same
    transform is reused for every frame in the batch. Keeping this transform
    fixed is important: later poses may move above, below, or beyond the root
    canvas, so they must expand the shared output canvas instead of changing
    earlier frame placement.
    """
    if match_axis not in {"Height (Y)", "Width (X)"}:
        raise ValueError(
            "match_axis must be 'Height (Y)' or 'Width (X)'; "
            f"received {match_axis!r}."
        )

    root_x, root_y, root_width, root_height = _foreground_bbox(root_frame)
    reference_x, reference_y, reference_width, reference_height = _foreground_bbox(
        reference_frame
    )

    scale_x = root_width / max(1.0, float(reference_width))
    scale_y = root_height / max(1.0, float(reference_height))
    root_center_x = root_x + root_width / 2.0
    reference_center_x = reference_x + reference_width / 2.0
    root_center_y = root_y + root_height / 2.0
    reference_center_y = reference_y + reference_height / 2.0

    if match_axis == "Height (Y)":
        translate_x = root_center_x - reference_center_x * scale_x
        translate_y = root_y - reference_y * scale_y
    else:
        translate_x = root_x - reference_x * scale_x
        translate_y = root_center_y - reference_center_y * scale_y

    return np.float32(
        (
            (scale_x, 0.0, translate_x),
            (0.0, scale_y, translate_y),
        )
    )


def _sample_distance(distance: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample a distance field at floating-point XY coordinates."""
    height, width = distance.shape
    valid = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < width - 1)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < height - 1)
    )
    points = points[valid]
    if len(points) == 0:
        return np.empty(0, dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    dx = x - x0
    dy = y - y0
    return (
        (1.0 - dx) * (1.0 - dy) * distance[y0, x0]
        + dx * (1.0 - dy) * distance[y0, x0 + 1]
        + (1.0 - dx) * dy * distance[y0 + 1, x0]
        + dx * dy * distance[y0 + 1, x0 + 1]
    )


def _edge_points(mask: np.ndarray) -> np.ndarray:
    """Return a deterministic, bounded sample of a silhouette edge."""
    edge = cv2.morphologyEx(
        mask,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    points = np.column_stack(np.where(edge > 0)[::-1]).astype(np.float32)
    if len(points) <= ALIGN_REFINEMENT_POINTS:
        return points

    indexes = np.linspace(
        0,
        len(points) - 1,
        ALIGN_REFINEMENT_POINTS,
        dtype=np.int32,
    )
    return points[indexes]


def _refine_alignment_transform(
    root_frame: np.ndarray,
    reference_frame: np.ndarray,
    initial_transform: np.ndarray,
) -> np.ndarray:
    """Fine-tune X/Y scale and translation against the two silhouettes.

    The search is deliberately local. Bounding-box matching establishes the
    stable animation scale first; this pass only corrects small registration
    errors such as a one- or two-pixel horizontal drift. Rotation and shear
    are excluded so pose changes cannot tilt or distort the batch.
    """
    root_mask = _foreground_mask(root_frame)
    reference_mask = _foreground_mask(reference_frame)
    root_distance = cv2.distanceTransform(
        (1 - root_mask).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    reference_distance = cv2.distanceTransform(
        (1 - reference_mask).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    root_points = _edge_points(root_mask)
    reference_points = _edge_points(reference_mask)
    if len(root_points) == 0 or len(reference_points) == 0:
        return initial_transform

    parameters = np.float32(
        (
            initial_transform[0, 0],
            initial_transform[1, 1],
            initial_transform[0, 2],
            initial_transform[1, 2],
        )
    )

    def score(candidate: np.ndarray) -> float:
        scale_x, scale_y, offset_x, offset_y = candidate
        if scale_x <= 0.0 or scale_y <= 0.0:
            return float("inf")

        transformed_reference = reference_points * np.float32(
            (scale_x, scale_y)
        ) + np.float32((offset_x, offset_y))
        forward = _sample_distance(root_distance, transformed_reference)

        transformed_root = (
            root_points - np.float32((offset_x, offset_y))
        ) / np.float32((scale_x, scale_y))
        backward = _sample_distance(reference_distance, transformed_root)
        if len(forward) == 0 or len(backward) == 0:
            return float("inf")
        return float(forward.mean() + backward.mean())

    best_score = score(parameters)
    # Do not fit pose changes as scale/translation. Fine registration is only
    # useful when frame 1 is already a close silhouette match to the root;
    # otherwise the bbox transform is the more stable animation anchor.
    if best_score > 12.0:
        return initial_transform

    initial = parameters.copy()
    for scale_step, offset_step in (
        (0.01, 2.0),
        (0.003, 0.5),
        (0.001, 0.15),
        (0.0003, 0.05),
    ):
        limits = np.float32(
            (
                (initial[0] * 0.98, initial[0] * 1.02),
                (initial[1] * 0.98, initial[1] * 1.02),
                (initial[2] - 8.0, initial[2] + 8.0),
                (initial[3] - 8.0, initial[3] + 8.0),
            )
        )
        for _ in range(8):
            changed = False
            for index, step in enumerate(
                (scale_step, scale_step, offset_step, offset_step)
            ):
                candidate_best = parameters
                candidate_score = best_score
                for direction in (-1.0, 1.0):
                    candidate = parameters.copy()
                    candidate[index] += direction * step
                    candidate[index] = np.clip(
                        candidate[index],
                        limits[index, 0],
                        limits[index, 1],
                    )
                    candidate_value = score(candidate)
                    if candidate_value < candidate_score:
                        candidate_best = candidate
                        candidate_score = candidate_value
                if candidate_score < best_score:
                    parameters = candidate_best
                    best_score = candidate_score
                    changed = True
            if not changed:
                break

    return np.float32(
        (
            (parameters[0], 0.0, parameters[2]),
            (0.0, parameters[1], parameters[3]),
        )
    )


def _transformed_bounds(
    transform: np.ndarray,
    frame: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return a frame's foreground bounds in root/output coordinates."""
    x, y, width, height = _foreground_bbox(frame)
    corners = np.float32(
        (
            (float(x), float(y), 1.0),
            (float(x + width), float(y), 1.0),
            (float(x), float(y + height), 1.0),
            (float(x + width), float(y + height), 1.0),
        )
    )
    target_corners = corners @ transform.T
    return (
        float(target_corners[:, 0].min()),
        float(target_corners[:, 1].min()),
        float(target_corners[:, 0].max()),
        float(target_corners[:, 1].max()),
    )


def _expanded_canvas(
    root_width: int,
    root_height: int,
    transform: np.ndarray,
    frames: np.ndarray,
) -> tuple[int, int, float, float]:
    """Build a shared canvas containing root and every transformed pose."""
    bounds = [(0.0, 0.0, float(root_width), float(root_height))]
    bounds.extend(_transformed_bounds(transform, frame) for frame in frames)
    min_x = min(bound[0] for bound in bounds) - ALIGN_PADDING_PX
    min_y = min(bound[1] for bound in bounds) - ALIGN_PADDING_PX
    max_x = max(bound[2] for bound in bounds) + ALIGN_PADDING_PX
    max_y = max(bound[3] for bound in bounds) + ALIGN_PADDING_PX

    canvas_min_x = float(np.floor(min_x))
    canvas_min_y = float(np.floor(min_y))
    canvas_max_x = float(np.ceil(max_x))
    canvas_max_y = float(np.ceil(max_y))
    return (
        max(1, int(canvas_max_x - canvas_min_x)),
        max(1, int(canvas_max_y - canvas_min_y)),
        -canvas_min_x,
        -canvas_min_y,
    )


def _translate_output_transform(
    transform: np.ndarray,
    offset_x: float,
    offset_y: float,
) -> np.ndarray:
    """Move a source-to-target warp into the expanded output canvas."""
    translated = transform.copy()
    translated[:, 2] += np.float32((offset_x, offset_y))
    return translated


class AlignFramesToRootNode:
    """Match a frame batch's subject scale and placement to a root image."""

    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("aligned_root", "aligned_frames")
    FUNCTION = "align"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_frame": (
                    "IMAGE",
                    {
                        "tooltip": "This reference image aligns every frame in the animation, preventing noticeable position and scale differences between animations—for example, between a standing idle and a jump-start animation. Choose a root frame that represents the animation’s intended ‘home position’.",
                    },
                ),
                "frames": (
                    "IMAGE",
                    {
                        "tooltip": "Frame batch to align. The correction is estimated from its first frame and applied to every frame.",
                    },
                ),
                "match_axis": (
                    ["Height (Y)", "Width (X)"],
                    {
                        "default": "Height (Y)",
                        "tooltip": "The first frame matches the root subject width and height independently. Height (Y) anchors their top edges and centers them on X; Width (X) anchors their left edges and centers them on Y.",
                    },
                ),
            },
        }

    def align(
        self,
        root_frame: torch.Tensor,
        frames: torch.Tensor,
        match_axis: str = "Height (Y)",
    ):
        root_frame = _as_image_batch(root_frame, "root_frame")
        frames = _as_image_batch(frames, "frames")

        if root_frame.shape[-1] != frames.shape[-1]:
            raise ValueError(
                "root_frame and frames must have the same channel count; "
                f"received {root_frame.shape[-1]} and {frames.shape[-1]}."
            )

        root_np = root_frame[0].detach().cpu().numpy().astype(np.float32, copy=False)
        frames_np = frames.detach().cpu().numpy().astype(np.float32, copy=False)
        transform = _alignment_transform(root_np, frames_np[0], match_axis)
        transform = _refine_alignment_transform(
            root_np,
            frames_np[0],
            transform,
        )
        output_width, output_height, offset_x, offset_y = _expanded_canvas(
            int(root_np.shape[1]),
            int(root_np.shape[0]),
            transform,
            frames_np,
        )
        output_transform = _translate_output_transform(
            transform,
            offset_x,
            offset_y,
        )

        root_background = _corner_background(root_np)
        root_border_value = root_background.tolist()
        if root_np.shape[-1] > 3:
            root_border_value.extend([1.0] * (root_np.shape[-1] - 3))
        root_transform = np.float32(
            (
                (1.0, 0.0, offset_x),
                (0.0, 1.0, offset_y),
            )
        )
        aligned_root = cv2.warpAffine(
            root_np,
            root_transform,
            (output_width, output_height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=root_border_value,
        )

        aligned = []
        for frame in frames_np:
            background = _corner_background(frame)
            border_value = background.tolist()
            if frame.shape[-1] > 3:
                border_value.extend([1.0] * (frame.shape[-1] - 3))
            aligned.append(
                cv2.warpAffine(
                    frame,
                    output_transform,
                    (output_width, output_height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=border_value,
                )
            )

        aligned_root_output = torch.from_numpy(aligned_root[None, ...]).to(
            device=frames.device,
            dtype=frames.dtype,
        )
        output = torch.from_numpy(np.stack(aligned, axis=0)).to(
            device=frames.device,
            dtype=frames.dtype,
        )
        return (aligned_root_output, output)
