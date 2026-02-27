"""Adjust VACE control_video strength without modifying core WanVaceToVideo."""

from __future__ import annotations


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
        new_meta = dict(meta)
        new_meta["vace_strength"] = [float(control_strength)]
        return new_meta

    @classmethod
    def _adjust_cond(cls, cond: list, control_strength: float) -> list:
        return [
            (tensor, cls._apply(meta, control_strength))
            for tensor, meta in cond
        ]

    def execute(self, positive, negative, control_strength):
        if abs(control_strength - 1.0) < 1e-6:
            return (positive, negative)

        pos = self._adjust_cond(positive, control_strength)
        neg = self._adjust_cond(negative, control_strength)
        return (pos, neg)
