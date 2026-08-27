from __future__ import annotations

import math


class AspectToResolution:
    """Calculate a multiple-aligned width and height from an aspect ratio."""

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate"
    CATEGORY = "utilities"

    ASPECT_RATIOS = {
        "1:1 (Square)": (1, 1),
        "2:3 (Portrait)": (2, 3),
        "3:2 (Landscape)": (3, 2),
        "4:5 (Portrait Standard)": (4, 5),
        "5:4 (Landscape Standard)": (5, 4),
        "9:16 (Portrait Widescreen)": (9, 16),
        "16:9 (Widescreen)": (16, 9),
        "1:2 (Tall)": (1, 2),
        "2:1 (Wide)": (2, 1),
        "9:21 (Very Tall Portrait)": (9, 21),
        "21:9 (Ultrawide)": (21, 9),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (list(cls.ASPECT_RATIOS),),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1},
                ),
                "multiple": (
                    "INT",
                    {"default": 8, "min": 8, "max": 128, "step": 4},
                ),
            }
        }

    def calculate(self, aspect_ratio: str, megapixels: float, multiple: int):
        width_ratio, height_ratio = self.ASPECT_RATIOS[aspect_ratio]
        total_pixels = megapixels * 1024 * 1024
        scale = math.sqrt(total_pixels / (width_ratio * height_ratio))
        width = round(width_ratio * scale / multiple) * multiple
        height = round(height_ratio * scale / multiple) * multiple
        return (width, height)
