from __future__ import annotations


class SpriteScaleCalculatorNode:
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "pixel_width",
        "pixel_height",
        "upscaled_width",
        "upscaled_height",
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
        "Spirie": {
            "width": _SPIRE_PRESET_WIDTH,
            "height": _SPIRE_PRESET_HEIGHT,
            "upscaled_width": 512,
            "upscaled_height": 1152,
        },
        # Custom dimensions currently use the same canvas until a separate
        # named character preset defines another canonical large size.
        "Custom": {
            "upscaled_width": 512,
            "upscaled_height": 1152,
        },
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
        preset_values = self._PRESETS.get(preset, self._PRESETS["Custom"])

        return (
            int(pixel_width),
            int(pixel_height),
            int(preset_values["upscaled_width"]),
            int(preset_values["upscaled_height"]),
        )
