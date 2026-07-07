from __future__ import annotations

import math


class SpriteScaleCalculatorNode:
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = (
        "pixel_width",
        "pixel_height",
    )
    FUNCTION = "calculate_pixels"
    CATEGORY = "image/transform"

    _SPIRE_PIXELS_PER_INCH = 2.0
    _SPIRE_PIXEL_WIDTH = 55
    _SPIRE_PIXEL_HEIGHT = 125
    _SPIRE_WIDTH_INCHES = _SPIRE_PIXEL_WIDTH / _SPIRE_PIXELS_PER_INCH
    _SPIRE_HEIGHT_INCHES = _SPIRE_PIXEL_HEIGHT / _SPIRE_PIXELS_PER_INCH
    _SPIRE_TARGET_WIDTH_INCHES = math.floor(_SPIRE_WIDTH_INCHES + 0.5)
    _SPIRE_TARGET_HEIGHT_INCHES = math.floor(_SPIRE_HEIGHT_INCHES + 0.5)

    _PRESETS = {
        "Spirie": (_SPIRE_TARGET_WIDTH_INCHES, _SPIRE_TARGET_HEIGHT_INCHES),
        "Custom": None,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls._PRESETS.keys()), {"default": "Spirie"}),
                "target_width_inches": (
                    "INT",
                    {"default": cls._SPIRE_TARGET_WIDTH_INCHES, "min": 0, "max": 10000, "step": 1},
                ),
                "target_height_inches": (
                    "INT",
                    {"default": cls._SPIRE_TARGET_HEIGHT_INCHES, "min": 0, "max": 10000, "step": 1},
                ),
            },
        }

    def calculate_pixels(
        self,
        preset: str,
        target_width_inches: int,
        target_height_inches: int,
    ):
        pixels_per_inch_width = self._SPIRE_PIXEL_WIDTH / self._SPIRE_WIDTH_INCHES
        pixels_per_inch_height = self._SPIRE_PIXEL_HEIGHT / self._SPIRE_HEIGHT_INCHES
        pixel_width = (
            0 if target_width_inches <= 0 else max(1, round(target_width_inches * pixels_per_inch_width))
        )
        pixel_height = (
            0 if target_height_inches <= 0 else max(1, round(target_height_inches * pixels_per_inch_height))
        )

        return (
            int(pixel_width),
            int(pixel_height),
        )
