"""Keep VACE control strength consistent as frame count increases."""

from __future__ import annotations


class VaceNormalizeStrength:
    """Normalize the control portion of expanded VACE strengths to a fixed baseline.

    This node is intended for conditioning produced by WanVace-style nodes that
    expand `vace_strength` into nested per-frame lists shaped like:

        [reference latent frames] + [control latent frames] (+ phantom frames)

    The goal is simple: `1.0` control strength on a short clip should not become
    effectively stronger just because a longer clip has more control latent-frame
    regions. To compensate, this node scales only the post-reference portion of
    the expanded strength list to match a fixed 9-frame baseline.

    WAN's 9-frame clip corresponds to 3 latent control frames, so the rule is:

        control_scale = 3 / actual_control_latent_frames

    Reference strengths are left unchanged.
    """

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    BASELINE_CONTROL_LATENTS = 3

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "trim_latent": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "tooltip": (
                            "Reference latent frame count prepended to each expanded "
                            "vace_strength list. Use the trim_latent output from "
                            "WanVacePhantomSimpleV2, or 0 when no reference image is used."
                        ),
                    },
                ),
            },
        }

    @classmethod
    def _normalize_control(cls, batch_strengths: list[float], trim_latent: int):
        ref_count = max(0, min(int(trim_latent), len(batch_strengths)))
        ref_part = list(batch_strengths[:ref_count])
        ctrl_part = list(batch_strengths[ref_count:])

        debug_info = {
            "ref_count": ref_count,
            "ctrl_count": len(ctrl_part),
            "scale": 1.0,
        }

        if not ctrl_part:
            return batch_strengths, debug_info

        scale = cls.BASELINE_CONTROL_LATENTS / float(len(ctrl_part))
        debug_info["scale"] = scale
        return ref_part + [value * scale for value in ctrl_part], debug_info

    @classmethod
    def _scale_meta(cls, meta: dict, trim_latent: int) -> tuple[dict, dict | None]:
        if "vace_strength" not in meta:
            return meta, None

        new_meta = dict(meta)
        vace_strength = new_meta["vace_strength"]
        new_vace_strength = []
        debug_info = None

        for context_strengths in vace_strength:
            if not isinstance(context_strengths, list):
                new_vace_strength.append(context_strengths)
                continue

            if context_strengths and not isinstance(context_strengths[0], list):
                scaled_batch, batch_debug = cls._normalize_control(context_strengths, trim_latent)
                if debug_info is None:
                    debug_info = batch_debug
                new_vace_strength.append(scaled_batch)
                continue

            new_context = []
            for batch_strengths in context_strengths:
                if not isinstance(batch_strengths, list):
                    new_context.append(batch_strengths)
                    continue
                scaled_batch, batch_debug = cls._normalize_control(batch_strengths, trim_latent)
                if debug_info is None:
                    debug_info = batch_debug
                new_context.append(scaled_batch)
            new_vace_strength.append(new_context)

        new_meta["vace_strength"] = new_vace_strength
        return new_meta, debug_info

    @classmethod
    def _adjust_cond(cls, cond: list, trim_latent: int) -> tuple[list, dict | None]:
        adjusted = []
        debug_info = None
        for tensor, meta in cond:
            new_meta, meta_debug = cls._scale_meta(meta, trim_latent)
            if debug_info is None and meta_debug is not None:
                debug_info = meta_debug
            adjusted.append((tensor, new_meta))
        return adjusted, debug_info

    def execute(self, positive, negative, trim_latent: int):
        pos, debug_info = self._adjust_cond(positive, trim_latent)
        neg, _ = self._adjust_cond(negative, trim_latent)

        try:
            if debug_info is not None:
                print(
                    "[VaceNormalizeStrength] "
                    f"baseline_control_latents={self.BASELINE_CONTROL_LATENTS}, "
                    f"control_scale={debug_info['scale']:.4f}, "
                    f"ref_count={debug_info['ref_count']}, ctrl_count={debug_info['ctrl_count']}"
                )
        except Exception:
            pass

        return (pos, neg)
