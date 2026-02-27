"""Adjust VACE control strength without modifying core WanVaceToVideo."""

from __future__ import annotations

import torch


class VaceControlStrength:
    """Split VACE conditioning so control_video strength can be adjusted separately.

    This expects conditioning produced by WanVaceToVideo, which stores a single
    vace_frames/vace_mask entry that may include reference frames at the start.
    Connect trim_latent from WanVaceToVideo to trim_amount to split reference
    frames (prefix) from control frames (suffix).
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
                    {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
                "reference_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
            }
        }

    @staticmethod
    def _split_vace(
        meta: dict, trim_amount: int, control_strength: float, reference_strength: float
    ) -> dict:
        if "vace_frames" not in meta or "vace_mask" not in meta:
            return meta

        vace_frames = meta.get("vace_frames")
        vace_mask = meta.get("vace_mask")

        if not isinstance(vace_frames, list) or not isinstance(vace_mask, list):
            return meta

        if len(vace_frames) != 1 or len(vace_mask) != 1:
            # Already split or unexpected format
            return meta

        frames = vace_frames[0]
        mask = vace_mask[0]
        if not isinstance(frames, torch.Tensor) or not isinstance(mask, torch.Tensor):
            return meta

        total = frames.shape[2]
        split = max(0, min(int(trim_amount), int(total)))

        new_meta = dict(meta)

        if split <= 0:
            new_meta["vace_frames"] = [frames]
            new_meta["vace_mask"] = [mask]
            new_meta["vace_strength"] = [float(control_strength)]
            return new_meta

        if split >= total:
            new_meta["vace_frames"] = [frames]
            new_meta["vace_mask"] = [mask]
            new_meta["vace_strength"] = [float(reference_strength)]
            return new_meta

        zeros = torch.zeros_like(frames)
        zeros_mask = torch.zeros_like(mask)

        ref_frames = zeros.clone()
        ref_mask = zeros_mask.clone()
        ref_frames[:, :, :split] = frames[:, :, :split]
        ref_mask[:, :, :split] = mask[:, :, :split]

        ctrl_frames = zeros.clone()
        ctrl_mask = zeros_mask.clone()
        ctrl_frames[:, :, split:] = frames[:, :, split:]
        ctrl_mask[:, :, split:] = mask[:, :, split:]

        new_meta["vace_frames"] = [ref_frames, ctrl_frames]
        new_meta["vace_mask"] = [ref_mask, ctrl_mask]
        new_meta["vace_strength"] = [float(reference_strength), float(control_strength)]
        return new_meta

    @classmethod
    def _adjust_cond(
        cls,
        cond: list,
        trim_amount: int,
        control_strength: float,
        reference_strength: float,
    ) -> list:
        out = []
        for entry in cond:
            tensor, meta = entry
            new_meta = cls._split_vace(
                meta, trim_amount, control_strength, reference_strength
            )
            out.append((tensor, new_meta))
        return out

    def execute(
        self, positive, negative, trim_amount, control_strength, reference_strength
    ):
        if (
            trim_amount <= 0
            and abs(control_strength - 1.0) < 1e-6
            and abs(reference_strength - 1.0) < 1e-6
        ):
            return (positive, negative)

        pos = self._adjust_cond(
            positive, trim_amount, control_strength, reference_strength
        )
        neg = self._adjust_cond(
            negative, trim_amount, control_strength, reference_strength
        )
        return (pos, neg)
