from __future__ import annotations

import numpy as np
import torch

from ...utils import parse_hex_color
from .palette_transfer import lab_to_rgb_u8, rgb_u8_to_lab


class MatchColorsToReferenceNode:
    """Match frame colors to a reference -- or to their own temporal consensus.

    Reinhard-style transfer in Lab: each frame's per-channel mean and
    standard deviation are matched to the destination's. A global shift/
    scale per channel, so the frame's internal color relationships are
    preserved exactly.

    frame_window pools per-frame Lab statistics over a centered window of
    neighboring frames. The window wraps around the batch (sequences are
    assumed to loop), so the last-to-first transition is stabilized like
    any other:
    - With image_ref connected, the pooled stats are the mapping's SOURCE
      side: frame t's mapping is built from the window consensus instead of
      its noisy solo stats, removing frame-to-frame jitter with no lag (the
      window is centered, unlike an EMA). Destination is the reference.
      window=1 = per-frame mapping.
    - Without image_ref, the pooled stats are the DESTINATION: each frame
      is pulled toward its neighbors' consensus. Smooths color drift with
      no reference needed, but only anchors locally -- drift longer than
      the window survives. window=1 = identity.

    Alpha awareness: statistics are computed from each frame alpha-
    composited onto background_color (a flat, per-frame-identical fill),
    not from the raw transparent pixels. A small, animating foreground
    (e.g. a keyed-out character) is too small and too frame-to-frame
    variable a sample on its own for a stable mean/std estimate --
    weighting the background out entirely (rather than filling it with a
    constant) throws away the large, stable sample that made per-frame
    statistics reliable in the first place, so the resulting mapping
    jitters wildly. Compositing onto a fixed color keeps the sample big
    and consistent across frames without that fill ever being visible:
    the resulting transform is applied to the frame's real content, and
    the frame's actual alpha (or embedded alpha channel) passes through
    to the output completely untouched.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched_frames",)
    FUNCTION = "match"
    CATEGORY = "color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Blend between original (0) and fully matched (1) colors.",
                    },
                ),
                "frame_window": (
                    "INT",
                    {
                        "default": 9,
                        "min": 1,
                        "max": 99,
                        "step": 2,
                        "tooltip": (
                            "Centered window of frames whose pooled Lab statistics "
                            "inform each frame's mapping (9 = itself + 4 behind + 4 "
                            "ahead; wraps around the batch, so loops stay seamless). "
                            "With image_ref connected it steadies the per-frame "
                            "mapping (1 = per-frame, no pooling); without image_ref "
                            "the pooled stats ARE the reference (1 = no-op). For "
                            "self-referential drift removal, use a window longer "
                            "than the drift."
                        ),
                    },
                ),
                "background_color": (
                    "STRING",
                    {
                        "default": "#FFFFFF",
                        "tooltip": (
                            "Flat fill used only to compute stable Lab statistics for "
                            "transparent/masked-out regions -- never appears in the "
                            "output, which keeps the frame's real alpha untouched. "
                            "Ignored for frames with no alpha channel and no mask."
                        ),
                    },
                ),
            },
            "optional": {
                "image_ref": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional absolute reference. Batch of 1 applies to all "
                            "targets; otherwise must match target batch size. When "
                            "unconnected, frames are matched toward their own "
                            "frame_window consensus instead."
                        )
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional per-pixel opacity for image_target used when "
                            "compositing onto background_color for statistics (e.g. "
                            "a keyer's mask, so a still-present chroma background "
                            "doesn't skew the mapping). White = real content, black "
                            "= background. Unconnected falls back to image_target's "
                            "own embedded alpha channel if present, otherwise the "
                            "frame is used as-is (nothing to composite)."
                        )
                    },
                ),
            },
        }

    # ------------------------------------------------------------------ util

    @staticmethod
    def _resolve_alpha(frames_u8: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
        """Per-frame opacity (N, H, W) in [0, 1] used only for background
        compositing: explicit mask, embedded alpha channel, or None if
        neither (nothing to composite -- use the frame's RGB as-is)."""
        if mask is not None:
            return mask
        if frames_u8.shape[-1] == 4:
            return frames_u8[..., 3].astype(np.float32) / 255.0
        return None

    @staticmethod
    def _composite_for_stats(
        frame_u8: np.ndarray, alpha: np.ndarray | None, bg_rgb: np.ndarray
    ) -> np.ndarray:
        """RGB (u8) of frame alpha-composited onto a flat background color,
        used only to compute stable statistics -- the real frame and its
        alpha are untouched anywhere else."""
        if alpha is None:
            return frame_u8[..., :3]
        rgb = frame_u8[..., :3].astype(np.float32)
        a = alpha[..., None]
        composited = rgb * a + bg_rgb[None, None, :] * (1.0 - a)
        return composited.round().clip(0, 255).astype(np.uint8)

    @staticmethod
    def _lab_stats(rgb_u8: np.ndarray) -> np.ndarray:
        """(2, 3) per-channel (mean, std) in Lab over every pixel."""
        lab = rgb_u8_to_lab(rgb_u8).reshape(-1, 3)
        mean = lab.mean(axis=0)
        var = ((lab - mean) ** 2).mean(axis=0)
        return np.stack([mean, np.sqrt(np.maximum(var, 0.0))])

    @staticmethod
    def _pool_stats(stats: list[np.ndarray], window: int) -> list[np.ndarray]:
        """Per-frame mean of stats over the centered window, wrapping around
        the batch (sequences are assumed to loop)."""
        n = len(stats)
        if window <= 1 or n <= 1:
            return stats
        radius = window // 2
        arr = np.stack(stats)
        return [
            arr[np.arange(t - radius, t + radius + 1) % n].mean(axis=0)
            for t in range(n)
        ]

    # ----------------------------------------------------------------- match

    def match(
        self,
        image_target: torch.Tensor,
        strength: float = 1.0,
        frame_window: int = 9,
        background_color: str = "#FFFFFF",
        image_ref: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        if strength <= 0.0:
            return (image_target,)

        tgt_np = (
            (image_target.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        )
        n = tgt_np.shape[0]

        tgt_mask_np = None
        if mask is not None:
            tgt_mask_np = mask.detach().cpu().numpy().astype(np.float32)
            if tgt_mask_np.ndim == 4 and tgt_mask_np.shape[-1] == 1:
                tgt_mask_np = tgt_mask_np[..., 0]
            if tgt_mask_np.shape[0] == 1 and n > 1:
                tgt_mask_np = np.repeat(tgt_mask_np, n, axis=0)
            if tgt_mask_np.shape[0] != n:
                raise ValueError(
                    f"Mask batch size ({tgt_mask_np.shape[0]}) must be 1 or match target batch size ({n})."
                )

        r, g, b = parse_hex_color(background_color, fallback=(255, 255, 255))
        bg_rgb = np.array([r, g, b], dtype=np.float32)

        tgt_alpha = self._resolve_alpha(tgt_np, tgt_mask_np)
        tgt_stats = [
            self._lab_stats(
                self._composite_for_stats(
                    tgt_np[i], tgt_alpha[i] if tgt_alpha is not None else None, bg_rgb
                )
            )
            for i in range(n)
        ]
        pooled = self._pool_stats(tgt_stats, frame_window)

        if image_ref is not None:
            ref_np = (
                (image_ref.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            )
            if ref_np.shape[0] not in (1, n):
                raise ValueError(
                    "Reference batch size must be 1 or match target batch size."
                )
            ref_alpha = self._resolve_alpha(ref_np, None)
            ref_stats = [
                self._lab_stats(
                    self._composite_for_stats(
                        ref_np[i], ref_alpha[i] if ref_alpha is not None else None, bg_rgb
                    )
                )
                for i in range(ref_np.shape[0])
            ]
            # Anchored mode: pooled window stats describe the frame (steady
            # source), the reference is the destination.
            sources = pooled
            dests = [ref_stats[0 if len(ref_stats) == 1 else i] for i in range(n)]
        else:
            # Self-referential mode: the frame's own stats are the source,
            # the window consensus is the destination.
            sources = tgt_stats
            dests = pooled

        results = []
        for idx in range(n):
            frame = tgt_np[idx]
            src_mean, src_std = sources[idx]
            dst_mean, dst_std = dests[idx]
            lab = rgb_u8_to_lab(frame[..., :3])
            scale = dst_std / np.maximum(src_std, 1e-6)
            lab = (lab - src_mean) * scale + dst_mean
            lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)
            lab[..., 1] = np.clip(lab[..., 1], -128.0, 127.0)
            lab[..., 2] = np.clip(lab[..., 2], -128.0, 127.0)
            matched = frame.copy()
            matched[..., :3] = lab_to_rgb_u8(lab)
            results.append(matched)

        matched_tensor = torch.from_numpy(np.stack(results)).float() / 255.0
        matched_tensor = matched_tensor.to(image_target.device)

        if strength < 1.0:
            matched_tensor = torch.lerp(image_target.float(), matched_tensor, strength)

        return (matched_tensor.clamp(0.0, 1.0),)
