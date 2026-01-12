from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import differential_evolution
from scipy.spatial import KDTree

from ..utils import (
    format_color_outputs,
    parse_color_value,
    rgb_to_hsv,
    rgb_to_int,
)
from .palette_transfer import detect_background_color, reinhard_transfer_lab


class ColorParserNode:
    """Parse a hex string or 24-bit integer into multiple representations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"multiline": False, "default": "3883558"}),
            }
        }

    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("24-bit", "hex", "rgb")
    FUNCTION = "parse_color"
    CATEGORY = "utils"

    def parse_color(self, value: str):
        return parse_color_value(value)


class FarthestColorNode:
    """Find a color farthest from the sampled pixels in an image batch."""

    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("24-bit", "hex", "rgb")
    FUNCTION = "find_farthest_color"
    CATEGORY = "utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "sample_rate": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "max_brightness": (
                    "INT",
                    {
                        "default": 140,
                        "min": 50,
                        "max": 200,
                        "step": 5,
                        "display": "number",
                    },
                ),
                "min_saturation": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.05,
                        "display": "number",
                    },
                ),
                "max_saturation": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.3,
                        "max": 0.9,
                        "step": 0.05,
                        "display": "number",
                    },
                ),
            },
        }

    def find_farthest_color(
        self,
        images: torch.Tensor,
        sample_rate: int = 10,
        max_brightness: int = 140,
        min_saturation: float = 0.2,
        max_saturation: float = 0.6,
    ):
        """
        Locate the color with the largest minimum distance from sampled image pixels,
        constrained to darker, muted tones.
        """
        images_np = images.detach().cpu().numpy()
        sampled_pixels = images_np[:, ::sample_rate, ::sample_rate, :3]
        pixels = (sampled_pixels.reshape(-1, 3) * 255.0).astype(np.float32)

        unique_pixels = np.unique(pixels, axis=0)
        tree = KDTree(unique_pixels)

        def objective(color: np.ndarray) -> float:
            r, g, b = color
            if not self._is_valid_color(
                r, g, b, max_brightness, min_saturation, max_saturation
            ):
                return 1e6
            distance, _ = tree.query(color)
            return -distance

        bounds = [(0, max_brightness)] * 3
        result = differential_evolution(
            objective,
            bounds,
            maxiter=150,
            popsize=20,
            seed=42,
            atol=0.01,
            tol=0.01,
        )

        r, g, b = np.clip(np.round(result.x).astype(int), 0, 255)
        if not self._is_valid_color(
            r, g, b, max_brightness, min_saturation, max_saturation
        ):
            r, g, b = (80, 100, 90)

        value_int = rgb_to_int(r, g, b)
        _, hex_str, rgb_str = format_color_outputs(value_int)

        distance_to_nearest, _ = tree.query(np.array([r, g, b], dtype=np.float32))
        h, s, v = rgb_to_hsv(r, g, b)
        print(
            f"[FarthestColorNode] Selected RGB({r}, {g}, {b}) / "
            f"HSV({h:.1f}°, {s:.2f}, {v:.2f}) - distance {distance_to_nearest:.2f}"
        )

        return value_int, hex_str, rgb_str

    @staticmethod
    def _is_valid_color(
        r: float,
        g: float,
        b: float,
        max_brightness: float,
        min_sat: float,
        max_sat: float,
    ) -> bool:
        h, s, v = rgb_to_hsv(r, g, b)
        brightness = v * 255.0
        return brightness <= max_brightness and min_sat <= s <= max_sat


class MatchColorPaletteNode:
    """Match target image colors to a reference palette using Reinhard grading."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graded_images",)
    FUNCTION = "match_color_palette"
    CATEGORY = "color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
            },
        }

    def match_color_palette(
        self,
        image_ref: torch.Tensor,
        image_target: torch.Tensor,
    ):
        no_bg_mask = False
        ref_np = (
            image_ref.detach().cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        tgt_np = (
            image_target.detach().cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)

        if ref_np.shape[0] not in (1, tgt_np.shape[0]):
            raise ValueError(
                "Reference batch size must be 1 or match target batch size."
            )

        results = []
        for idx in range(tgt_np.shape[0]):
            ref_idx = 0 if ref_np.shape[0] == 1 else idx
            ref_rgb, ref_alpha = self._split_alpha(ref_np[ref_idx])
            tgt_rgb, tgt_alpha = self._split_alpha(tgt_np[idx])

            reference_mask = ref_alpha > 0 if ref_alpha is not None else None
            target_mask_alpha = tgt_alpha > 0 if tgt_alpha is not None else None

            if no_bg_mask:
                reference_content_mask = (
                    reference_mask
                    if reference_mask is not None
                    else np.ones(ref_rgb.shape[:2], dtype=bool)
                )
                target_content_mask = np.ones(tgt_rgb.shape[:2], dtype=bool)
                if target_mask_alpha is not None:
                    target_content_mask &= target_mask_alpha
            else:
                bg_rgb = detect_background_color(tgt_rgb)
                if reference_mask is not None:
                    reference_content_mask = reference_mask & np.any(
                        ref_rgb != bg_rgb, axis=-1
                    )
                else:
                    reference_content_mask = np.any(ref_rgb != bg_rgb, axis=-1)
                target_content_mask = np.any(tgt_rgb != bg_rgb, axis=-1)
                if target_mask_alpha is not None:
                    target_content_mask &= target_mask_alpha

            graded_rgb = reinhard_transfer_lab(
                tgt_rgb,
                ref_rgb,
                target_mask=target_content_mask,
                reference_mask=reference_content_mask,
            )

            if tgt_alpha is not None:
                graded = np.dstack((graded_rgb, tgt_alpha))
            else:
                graded = graded_rgb
            results.append(graded.astype(np.uint8))

        result_tensor = torch.from_numpy(np.stack(results)).float() / 255.0
        return (result_tensor,)

    @staticmethod
    def _split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if image.shape[-1] == 4:
            return image[:, :, :3], image[:, :, 3]
        return image[:, :, :3], None
