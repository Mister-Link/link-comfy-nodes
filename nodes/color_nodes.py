from __future__ import annotations

import re

import numpy as np
import torch
from scipy.optimize import differential_evolution
from scipy.spatial import KDTree

from ..utils import (
    format_color_outputs,
    parse_color_value,
    parse_hex_color,
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


class PaletteMatchNode:
    """Force a target image batch to use only colors from a master palette."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched_images",)
    FUNCTION = "match_palette"
    CATEGORY = "color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_images": ("IMAGE",),
                "target_images": ("IMAGE",),
            },
            "optional": {
                "sample_rate": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 20,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "max_colors": (
                    "INT",
                    {
                        "default": 256,
                        "min": 2,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "include_target_in_palette": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "merge_distance": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "use_lab_distance": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "ignore_color": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
            },
        }

    def match_palette(
        self,
        master_images: torch.Tensor,
        target_images: torch.Tensor,
        sample_rate: int = 1,
        max_colors: int = 256,
        include_target_in_palette: bool = False,
        merge_distance: int = 0,
        use_lab_distance: bool = False,
        ignore_color: str = "",
    ):
        ignore_rgb = self._parse_optional_hex(ignore_color)
        palette, counts = self._build_palette(
            master_images, target_images, sample_rate, include_target_in_palette
        )
        if ignore_rgb is not None:
            keep = np.any(palette != np.array(ignore_rgb, dtype=np.uint8), axis=1)
            palette = palette[keep]
            counts = counts[keep]

        if merge_distance > 0:
            palette, counts = self._merge_nearby_colors(
                palette, counts, merge_distance, use_lab_distance
            )

        if palette.shape[0] > max_colors:
            top_idx = np.argsort(counts)[-max_colors:]
            palette = palette[top_idx]

        if palette.size == 0:
            raise ValueError("Master palette is empty after filtering.")

        if use_lab_distance:
            palette_space = self._rgb_to_lab(palette)
        else:
            palette_space = palette.astype(np.float32)
        tree = KDTree(palette_space)
        target_np = (
            target_images.detach().cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)

        remapped = []
        for img in target_np:
            pixels = img.reshape(-1, 3)
            if use_lab_distance:
                pixels_space = self._rgb_to_lab(pixels)
            else:
                pixels_space = pixels.astype(np.float32)
            _, idx = tree.query(pixels_space)
            mapped_pixels = palette[idx].astype(np.uint8)
            remapped.append(mapped_pixels.reshape(img.shape))

        result = torch.from_numpy(np.stack(remapped)).float() / 255.0
        return (result,)

    @staticmethod
    def _build_palette(
        master_images: torch.Tensor,
        target_images: torch.Tensor,
        sample_rate: int,
        include_target_in_palette: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        def sample_pixels(images: torch.Tensor) -> np.ndarray:
            images_np = (
                images.detach().cpu().numpy() * 255.0
            ).round().clip(0, 255).astype(np.uint8)
            sampled = images_np[:, ::sample_rate, ::sample_rate, :3]
            return sampled.reshape(-1, 3)

        pixels = sample_pixels(master_images)
        if include_target_in_palette:
            target_pixels = sample_pixels(target_images)
            pixels = np.concatenate([pixels, target_pixels], axis=0)

        palette, counts = np.unique(pixels, axis=0, return_counts=True)
        return palette, counts

    @staticmethod
    def _merge_nearby_colors(
        palette: np.ndarray,
        counts: np.ndarray,
        merge_distance: int,
        use_lab_distance: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if palette.size == 0:
            return palette, counts

        order = np.argsort(counts)[::-1]
        merged_colors: list[np.ndarray] = []
        merged_space: list[np.ndarray] = []
        merged_counts: list[float] = []

        if use_lab_distance:
            palette_space = PaletteMatchNode._rgb_to_lab(palette)
        else:
            palette_space = palette.astype(np.float32)

        for idx in order:
            color = palette[idx].astype(np.float32)
            color_space = palette_space[idx]
            if merged_space:
                dist = np.linalg.norm(np.stack(merged_space) - color_space, axis=1)
                nearest = int(np.argmin(dist))
                if dist[nearest] <= float(merge_distance):
                    total = merged_counts[nearest] + float(counts[idx])
                    merged_colors[nearest] = (
                        merged_colors[nearest] * merged_counts[nearest]
                        + color * float(counts[idx])
                    ) / total
                    merged_counts[nearest] = total
                    if use_lab_distance:
                        merged_space[nearest] = PaletteMatchNode._rgb_to_lab(
                            np.round(merged_colors[nearest]).astype(np.uint8)[None, :]
                        )[0]
                    continue
            merged_colors.append(color)
            merged_space.append(color_space)
            merged_counts.append(float(counts[idx]))

        merged = np.clip(np.round(merged_colors), 0, 255).astype(np.uint8)
        if merged.size == 0:
            return merged, np.array([], dtype=np.float32)

        unique, inverse = np.unique(merged, axis=0, return_inverse=True)
        weights = np.zeros(unique.shape[0], dtype=np.float32)
        for idx, weight in enumerate(merged_counts):
            weights[inverse[idx]] += float(weight)
        return unique, weights

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        rgb_f = rgb.astype(np.float32) / 255.0
        linear = np.where(
            rgb_f <= 0.04045,
            rgb_f / 12.92,
            ((rgb_f + 0.055) / 1.055) ** 2.4,
        )
        x = linear[..., 0] * 0.4124 + linear[..., 1] * 0.3576 + linear[..., 2] * 0.1805
        y = linear[..., 0] * 0.2126 + linear[..., 1] * 0.7152 + linear[..., 2] * 0.0722
        z = linear[..., 0] * 0.0193 + linear[..., 1] * 0.1192 + linear[..., 2] * 0.9505

        x /= 0.95047
        z /= 1.08883

        epsilon = 0.008856
        kappa = 903.3

        fx = np.where(x > epsilon, np.cbrt(x), (kappa * x + 16.0) / 116.0)
        fy = np.where(y > epsilon, np.cbrt(y), (kappa * y + 16.0) / 116.0)
        fz = np.where(z > epsilon, np.cbrt(z), (kappa * z + 16.0) / 116.0)

        l = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return np.stack([l, a, b], axis=-1).astype(np.float32)

    @staticmethod
    def _parse_optional_hex(value: str):
        if not value:
            return None
        value = value.strip()
        if not re.match(r"^#?[0-9a-fA-F]{1,6}$", value):
            return None
        return parse_hex_color(value)
