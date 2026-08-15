from __future__ import annotations


class SpriteScaleCalculatorNode:
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = (
        "pixel_width",
        "pixel_height",
    )
    FUNCTION = "calculate_pixels"
    CATEGORY = "image/transform"

    # "width"/"height" are an arbitrary world unit, not pixels directly --
    # pixel_width/pixel_height scale them up by this factor. Spirie is 5'0"
    # (60") tall, so the preset's height=80 units works out to 60/80 = 0.75
    # inches/unit for the feet/inches preview label (web/spriteScaleCalculator.js);
    # that calibration is independent of this pixel scale factor.
    _PIXELS_PER_UNIT = 2.0
    _SPIRE_PRESET_WIDTH = 35
    _SPIRE_PRESET_HEIGHT = 80

    _PRESETS = {
        "Spirie": (_SPIRE_PRESET_WIDTH, _SPIRE_PRESET_HEIGHT),
        "Custom": None,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls._PRESETS.keys()), {"default": "Spirie"}),
                "width": (
                    "INT",
                    {"default": cls._SPIRE_PRESET_WIDTH, "min": 0, "max": 10000, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": cls._SPIRE_PRESET_HEIGHT, "min": 0, "max": 10000, "step": 1},
                ),
            },
        }

    def calculate_pixels(
        self,
        preset: str,
        width: int,
        height: int,
    ):
        pixel_width = 0 if width <= 0 else max(1, round(width * self._PIXELS_PER_UNIT))
        pixel_height = 0 if height <= 0 else max(1, round(height * self._PIXELS_PER_UNIT))

        return (
            int(pixel_width),
            int(pixel_height),
        )
