"""Balance VACE reference and control strengths without patching WanVace nodes."""

from __future__ import annotations

import math


class VaceNormalizeStrength:
    """Rebalance expanded VACE strength lists for WanVace-style conditioning.

    This node is intended for conditioning produced by nodes that expand
    ``vace_strength`` into nested per-frame lists, such as
    ``WanVacePhantomSimpleV2``. In that format the strength list is typically:

        [reference latent frames] + [control latent frames] (+ [phantom frames])

    The practical issue is usually not that reference or control is missing from
    the model's per-block application. Both are applied every VACE block. The
    imbalance comes from one side covering many more latent-frame regions than
    the other.

    This node therefore focuses on balancing total reference/control influence by
    scaling one side of the expanded list while preserving its internal shape.
    Legacy length-normalization modes are kept for existing workflows.
    """

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    BALANCE_MODES = [
        "reference_sqrt_match_control",
        "reference_match_control",
        "reference_target_ratio",
        "control_sqrt_match_reference",
        "control_match_reference",
        "legacy_control_sqrt_reference_frames",
        "legacy_control_linear_reference_frames",
    ]

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
                "mode": (
                    cls.BALANCE_MODES,
                    {
                        "default": "reference_sqrt_match_control",
                        "tooltip": (
                            "How to rebalance the expanded VACE strengths. The default "
                            "gently boosts reference mass toward control mass. Legacy "
                            "modes preserve the old length-only control normalization."
                        ),
                    },
                ),
                "target_ratio": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": (
                            "Used only by reference_target_ratio. Desired total "
                            "reference mass divided by total control mass after scaling."
                        ),
                    },
                ),
                "max_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": (
                            "Clamp for the computed multiplicative scale to avoid "
                            "extreme boosts on very long clips."
                        ),
                    },
                ),
                "reference_frames": (
                    "INT",
                    {
                        "default": 9,
                        "min": 1,
                        "max": 10000,
                        "step": 4,
                        "tooltip": (
                            "Used only by legacy modes. Video frame count at which the "
                            "control strength was calibrated."
                        ),
                    },
                ),
            },
        }

    @staticmethod
    def _clamp_scale(scale: float, max_scale: float) -> float:
        return max(0.0, min(float(scale), float(max_scale)))

    @staticmethod
    def _legacy_scale(actual_control_latent: int, reference_frames: int, mode: str) -> float:
        ref_latent = ((int(reference_frames) - 1) // 4) + 1
        if actual_control_latent <= 0 or ref_latent <= 0:
            return 1.0
        if mode == "legacy_control_sqrt_reference_frames":
            return math.sqrt(ref_latent / actual_control_latent)
        return ref_latent / actual_control_latent

    @staticmethod
    def _compute_scale(
        ref_part: list[float],
        ctrl_part: list[float],
        mode: str,
        target_ratio: float,
        reference_frames: int,
    ) -> tuple[str, float]:
        ref_total = float(sum(ref_part))
        ctrl_total = float(sum(ctrl_part))
        ctrl_count = len(ctrl_part)

        if mode.startswith("legacy_"):
            return ("control", VaceNormalizeStrength._legacy_scale(ctrl_count, reference_frames, mode))

        if not ref_part or not ctrl_part:
            return ("none", 1.0)

        if mode == "reference_sqrt_match_control":
            if ref_total <= 0.0 or ctrl_total <= 0.0:
                return ("none", 1.0)
            return ("reference", math.sqrt(ctrl_total / ref_total))

        if mode == "reference_match_control":
            if ref_total <= 0.0 or ctrl_total <= 0.0:
                return ("none", 1.0)
            return ("reference", ctrl_total / ref_total)

        if mode == "reference_target_ratio":
            if ref_total <= 0.0 or ctrl_total <= 0.0:
                return ("none", 1.0)
            return ("reference", (ctrl_total * float(target_ratio)) / ref_total)

        if mode == "control_sqrt_match_reference":
            if ref_total <= 0.0 or ctrl_total <= 0.0:
                return ("none", 1.0)
            return ("control", math.sqrt(ref_total / ctrl_total))

        if mode == "control_match_reference":
            if ref_total <= 0.0 or ctrl_total <= 0.0:
                return ("none", 1.0)
            return ("control", ref_total / ctrl_total)

        return ("none", 1.0)

    @classmethod
    def _scale_batch_strengths(
        cls,
        batch_strengths,
        trim_latent: int,
        mode: str,
        target_ratio: float,
        max_scale: float,
        reference_frames: int,
    ):
        if not isinstance(batch_strengths, list):
            return batch_strengths, None

        ref_count = max(0, min(int(trim_latent), len(batch_strengths)))
        ref_part = list(batch_strengths[:ref_count])
        ctrl_part = list(batch_strengths[ref_count:])

        target, scale = cls._compute_scale(
            ref_part,
            ctrl_part,
            mode,
            target_ratio,
            reference_frames,
        )
        scale = cls._clamp_scale(scale, max_scale)

        if target == "reference" and ref_part:
            return ([value * scale for value in ref_part] + ctrl_part, {
                "target": target,
                "scale": scale,
                "ref_count": ref_count,
                "ctrl_count": len(ctrl_part),
                "ref_total_before": sum(ref_part),
                "ctrl_total_before": sum(ctrl_part),
            })

        if target == "control" and ctrl_part:
            return (ref_part + [value * scale for value in ctrl_part], {
                "target": target,
                "scale": scale,
                "ref_count": ref_count,
                "ctrl_count": len(ctrl_part),
                "ref_total_before": sum(ref_part),
                "ctrl_total_before": sum(ctrl_part),
            })

        return (batch_strengths, {
            "target": "none",
            "scale": 1.0,
            "ref_count": ref_count,
            "ctrl_count": len(ctrl_part),
            "ref_total_before": sum(ref_part),
            "ctrl_total_before": sum(ctrl_part),
        })

    @classmethod
    def _scale_meta(
        cls,
        meta: dict,
        trim_latent: int,
        mode: str,
        target_ratio: float,
        max_scale: float,
        reference_frames: int,
    ) -> tuple[dict, dict | None]:
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
                scaled_batch, batch_debug = cls._scale_batch_strengths(
                    context_strengths,
                    trim_latent,
                    mode,
                    target_ratio,
                    max_scale,
                    reference_frames,
                )
                if debug_info is None and batch_debug is not None:
                    debug_info = batch_debug
                new_vace_strength.append(scaled_batch)
                continue

            new_context = []
            for batch_strengths in context_strengths:
                scaled_batch, batch_debug = cls._scale_batch_strengths(
                    batch_strengths,
                    trim_latent,
                    mode,
                    target_ratio,
                    max_scale,
                    reference_frames,
                )
                if debug_info is None and batch_debug is not None:
                    debug_info = batch_debug
                new_context.append(scaled_batch)
            new_vace_strength.append(new_context)

        new_meta["vace_strength"] = new_vace_strength
        return new_meta, debug_info

    @classmethod
    def _adjust_cond(
        cls,
        cond: list,
        trim_latent: int,
        mode: str,
        target_ratio: float,
        max_scale: float,
        reference_frames: int,
    ) -> tuple[list, dict | None]:
        adjusted = []
        debug_info = None
        for tensor, meta in cond:
            new_meta, meta_debug = cls._scale_meta(
                meta,
                trim_latent,
                mode,
                target_ratio,
                max_scale,
                reference_frames,
            )
            if debug_info is None and meta_debug is not None:
                debug_info = meta_debug
            adjusted.append((tensor, new_meta))
        return adjusted, debug_info

    def execute(
        self,
        positive,
        negative,
        trim_latent: int,
        mode: str = "reference_sqrt_match_control",
        target_ratio: float = 1.0,
        max_scale: float = 4.0,
        reference_frames: int = 9,
    ):
        pos, debug_info = self._adjust_cond(
            positive,
            trim_latent,
            mode,
            target_ratio,
            max_scale,
            reference_frames,
        )
        neg, _ = self._adjust_cond(
            negative,
            trim_latent,
            mode,
            target_ratio,
            max_scale,
            reference_frames,
        )

        try:
            if debug_info is not None:
                print(
                    "[VaceNormalizeStrength] "
                    f"mode={mode}, target={debug_info['target']}, "
                    f"scale={debug_info['scale']:.4f}, "
                    f"ref_count={debug_info['ref_count']}, ctrl_count={debug_info['ctrl_count']}, "
                    f"ref_total={debug_info['ref_total_before']:.4f}, "
                    f"ctrl_total={debug_info['ctrl_total_before']:.4f}"
                )
        except Exception:
            pass

        return (pos, neg)
