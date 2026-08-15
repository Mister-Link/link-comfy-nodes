from __future__ import annotations

import numpy as np
import torch

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

    Alpha awareness: an embedded 4th channel weights the statistics, so
    transparent regions (e.g. a keyed-out chroma background still present
    in RGB) don't skew the mapping. The target's alpha channel is joined
    back onto the output untouched.
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
            },
        }

    # ------------------------------------------------------------------ util

    @staticmethod
    def _resolve_weights(frames_u8: np.ndarray) -> np.ndarray:
        """Per-frame opacity weights (N, H, W): embedded alpha or ones."""
        if frames_u8.shape[-1] == 4:
            return frames_u8[..., 3].astype(np.float32) / 255.0
        return np.ones(frames_u8.shape[:3], dtype=np.float32)

    @staticmethod
    def _weighted_lab_stats(frame_u8: np.ndarray, w: np.ndarray) -> np.ndarray:
        """(2, 3) alpha-weighted per-channel (mean, std) in Lab."""
        lab = rgb_u8_to_lab(frame_u8[..., :3]).reshape(-1, 3)
        wr = w.reshape(-1, 1)
        total = max(wr.sum(), 1e-12)
        mean = (lab * wr).sum(axis=0) / total
        var = ((lab - mean) ** 2 * wr).sum(axis=0) / total
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
        image_ref: torch.Tensor | None = None,
    ):
        if strength <= 0.0:
            return (image_target,)

        tgt_np = (
            (image_target.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        )
        tgt_w = self._resolve_weights(tgt_np)
        n = tgt_np.shape[0]

        tgt_stats = [self._weighted_lab_stats(tgt_np[i], tgt_w[i]) for i in range(n)]
        pooled = self._pool_stats(tgt_stats, frame_window)

        if image_ref is not None:
            ref_np = (
                (image_ref.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            )
            if ref_np.shape[0] not in (1, n):
                raise ValueError(
                    "Reference batch size must be 1 or match target batch size."
                )
            ref_w = self._resolve_weights(ref_np)
            ref_stats = [
                self._weighted_lab_stats(ref_np[i], ref_w[i])
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
