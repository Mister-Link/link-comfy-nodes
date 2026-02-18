"""Trim VACE conditioning to match a trimmed latent."""

from __future__ import annotations

import torch


class TrimConditioning:
    """
    Trims the temporal dimension of VACE conditioning to match a TrimVideoLatent output.

    WanVaceToVideo prepends reference image frames to the latent and conditioning
    when reference_image is used.  TrimVideoLatent removes those frames from the
    latent; this node removes them from positive/negative conditioning so the
    frame counts stay in sync.

    Slices vace_frames and vace_mask by trim_amount on their temporal axis (dim 2).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "trim_amount": ("INT", {"default": 0, "min": 0, "max": 1024}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Trims VACE conditioning temporal frames to match TrimVideoLatent output. "
        "Connect trim_latent from WanVaceToVideo to trim_amount."
    )

    @staticmethod
    def _trim_cond(cond: list, trim_amount: int) -> list:
        if trim_amount <= 0:
            return cond

        out = []
        for entry in cond:
            tensor, meta = entry
            new_meta = dict(meta)

            if "vace_frames" in new_meta:
                new_meta["vace_frames"] = [
                    f[:, :, trim_amount:, :, :] for f in new_meta["vace_frames"]
                ]

            if "vace_mask" in new_meta:
                new_meta["vace_mask"] = [
                    m[:, :, trim_amount:, :, :] for m in new_meta["vace_mask"]
                ]

            out.append((tensor, new_meta))
        return out

    def execute(self, positive, negative, trim_amount):
        if trim_amount <= 0:
            return (positive, negative)

        print(
            f"[Trim VACE Conditioning] Trimming {trim_amount} frames from vace_frames/vace_mask"
        )
        pos = self._trim_cond(positive, trim_amount)
        neg = self._trim_cond(negative, trim_amount)
        return (pos, neg)


__all__ = ["TrimConditioning"]
