"""Align a frame batch to the scale and placement of a root frame."""

from __future__ import annotations

import cv2
import numpy as np
import torch


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
) -> np.ndarray:
    """Estimate a root-relative warp for the reference frame.

    ECC returns a matrix used with WARP_INVERSE_MAP: it maps output/root
    coordinates back into the reference image. Starting from the bbox
    transform lets this converge even when the two subjects have different
    scales or offsets; the silhouette refinement fixes cases where their
    bounding boxes happen to be identical.
    """
    root_mask = _foreground_mask(root_frame)
    reference_mask = _foreground_mask(reference_frame)

    root_x, root_y, root_width, root_height = _foreground_bbox(root_frame)
    reference_x, reference_y, reference_width, reference_height = _foreground_bbox(
        reference_frame
    )

    scale_x = root_width / max(1.0, float(reference_width))
    scale_y = root_height / max(1.0, float(reference_height))
    source_to_target = np.float32(
        (
            (scale_x, 0.0, root_x - (reference_x * scale_x)),
            (0.0, scale_y, root_y - (reference_y * scale_y)),
        )
    )
    fallback = cv2.invertAffineTransform(source_to_target)

    try:
        _, refined = cv2.findTransformECC(
            root_mask.astype(np.float32),
            reference_mask.astype(np.float32),
            fallback.copy(),
            cv2.MOTION_AFFINE,
            (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                200,
                1e-6,
            ),
            None,
            1,
        )
        if np.all(np.isfinite(refined)):
            return refined.astype(np.float32, copy=False)
    except cv2.error:
        pass

    return fallback


class AlignFramesToRootNode:
    """Match a frame batch's subject scale and placement to a root image."""

    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("aligned_frames",)
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
            },
        }

    def align(self, root_frame: torch.Tensor, frames: torch.Tensor):
        root_frame = _as_image_batch(root_frame, "root_frame")
        frames = _as_image_batch(frames, "frames")

        if root_frame.shape[-1] != frames.shape[-1]:
            raise ValueError(
                "root_frame and frames must have the same channel count; "
                f"received {root_frame.shape[-1]} and {frames.shape[-1]}."
            )

        root_np = root_frame[0].detach().cpu().numpy().astype(np.float32, copy=False)
        frames_np = frames.detach().cpu().numpy().astype(np.float32, copy=False)
        transform = _alignment_transform(root_np, frames_np[0])

        aligned = []
        for frame in frames_np:
            background = _corner_background(frame)
            border_value = background.tolist()
            if frame.shape[-1] > 3:
                border_value.extend([1.0] * (frame.shape[-1] - 3))
            aligned.append(
                cv2.warpAffine(
                    frame,
                    transform,
                    (int(root_np.shape[1]), int(root_np.shape[0])),
                    flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=border_value,
                )
            )

        output = torch.from_numpy(np.stack(aligned, axis=0)).to(
            device=frames.device,
            dtype=frames.dtype,
        )
        return (output,)
