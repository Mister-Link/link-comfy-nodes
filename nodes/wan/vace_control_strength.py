"""Adjust VACE control_video strength without modifying core WanVaceToVideo."""

from __future__ import annotations

import torch


class VaceControlStrength:
    """Scale VACE control_video strength via the vace_strength scalar.

    Sets vace_strength in the conditioning so the model applies it as a
    multiplier on the VACE skip-connection output (x += c_skip * vace_strength).
    This is the correct way to attenuate control influence without corrupting
    the latent values, which would happen if the VAE-encoded frames were
    scaled directly (they go through process_latent_in normalisation inside
    extra_conds, which does not commute with a pre-scale).

    WanVaceToVideo sets vace_strength=1.0 by default; use values below 1.0
    to reduce the influence of control_video on the generated output.
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
                "control_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    @staticmethod
    def _apply(meta: dict, control_strength: float) -> dict:
        if "vace_frames" not in meta:
            return meta

        control_strength = float(control_strength)
        new_meta = dict(meta)
        new_meta["vace_strength"] = [control_strength]

        vace_frames = new_meta.get("vace_frames")
        vace_mask = new_meta.get("vace_mask")

        # Blend reactive channels (16:32) toward inactive channels (0:16).
        # This mirrors the model-patch approach and suppresses pose skeleton
        # imprinting that can persist in long sequences.
        if isinstance(vace_frames, (list, tuple)):
            blended_frames = []
            for idx, frames in enumerate(vace_frames):
                if (
                    not torch.is_tensor(frames)
                    or frames.ndim < 2
                    or frames.shape[1] < 32
                ):
                    blended_frames.append(frames)
                    continue

                inactive = frames[:, 0:16]
                reactive = frames[:, 16:32]
                frame_strength: float | torch.Tensor = control_strength

                if (
                    isinstance(vace_mask, (list, tuple))
                    and idx < len(vace_mask)
                    and torch.is_tensor(vace_mask[idx])
                ):
                    mask = vace_mask[idx]
                    if (
                        mask.ndim == frames.ndim == 5
                        and mask.shape[0] == frames.shape[0]
                        and mask.shape[2] == frames.shape[2]
                    ):
                        # Disable attenuation on reference-only temporal slots
                        # (where mask is entirely zero), keep it on control slots.
                        frame_gate = (
                            mask.abs().amax(dim=(1, 3, 4), keepdim=True) > 0
                        ).to(dtype=frames.dtype)
                        frame_strength = 1.0 - ((1.0 - control_strength) * frame_gate)

                frames_out = frames.clone()
                frames_out[:, 16:32] = inactive + (
                    (reactive - inactive) * frame_strength
                )
                blended_frames.append(frames_out)

            new_meta["vace_frames"] = blended_frames

        # Attenuate vace_mask alongside vace_strength. With pose controls such
        # as OpenPose, keeping masks at full scale can still imprint skeleton
        # structure in longer clips even when strength is reduced.
        if isinstance(vace_mask, (list, tuple)):
            scaled_mask = []
            for mask in vace_mask:
                if torch.is_tensor(mask):
                    scaled_mask.append(mask * control_strength)
                else:
                    scaled_mask.append(mask)
            new_meta["vace_mask"] = scaled_mask

        return new_meta

    @classmethod
    def _adjust_cond(cls, cond: list, control_strength: float) -> list:
        return [(tensor, cls._apply(meta, control_strength)) for tensor, meta in cond]

    def execute(self, positive, negative, control_strength):
        if abs(control_strength - 1.0) < 1e-6:
            return (positive, negative)

        pos = self._adjust_cond(positive, control_strength)
        neg = self._adjust_cond(negative, control_strength)
        return (pos, neg)
