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
