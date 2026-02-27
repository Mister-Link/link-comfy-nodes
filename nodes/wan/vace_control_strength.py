"""Adjust VACE control_video strength without modifying core WanVaceToVideo."""

from __future__ import annotations

import torch


class VaceControlStrength:
    """Scale VACE control_video strength independently of the reference_image.

    This expects conditioning produced by WanVaceToVideo, which stores a single
    vace_frames/vace_mask entry that may include reference frames at the start.
    Connect trim_latent from WanVaceToVideo to trim_amount so the node knows
    where reference frames end and control frames begin. Reference frames are
    left untouched; only the control_video portion is scaled.
    """

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "trim_amount": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "control_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    @staticmethod
    def _apply(meta: dict, trim_amount: int, control_strength: float) -> dict:
        if "vace_frames" not in meta or "vace_mask" not in meta:
            return meta

        vace_frames = meta.get("vace_frames")
        vace_mask = meta.get("vace_mask")

        if not isinstance(vace_frames, list) or not isinstance(vace_mask, list):
            return meta

        if len(vace_frames) != 1 or len(vace_mask) != 1:
            return meta

        frames = vace_frames[0]
        mask = vace_mask[0]
        if not isinstance(frames, torch.Tensor) or not isinstance(mask, torch.Tensor):
            return meta

        total = frames.shape[2]
        split = max(0, min(int(trim_amount), int(total)))

        new_meta = dict(meta)
        frames_out = frames.clone()

        # Only scale the control_video portion (after the reference prefix).
        frames_out[:, :, split:] *= float(control_strength)

        new_meta["vace_frames"] = [frames_out]
        # Strength is now baked into the frames; keep the scalar at 1.
        new_meta["vace_strength"] = [1.0]
        return new_meta

    @classmethod
    def _adjust_cond(cls, cond: list, trim_amount: int, control_strength: float) -> list:
        return [
            (tensor, cls._apply(meta, trim_amount, control_strength))
            for tensor, meta in cond
        ]

    def execute(self, positive, negative, trim_amount, control_strength):
        if abs(control_strength - 1.0) < 1e-6:
            return (positive, negative)

        pos = self._adjust_cond(positive, trim_amount, control_strength)
        neg = self._adjust_cond(negative, trim_amount, control_strength)
        return (pos, neg)
