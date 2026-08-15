from __future__ import annotations


class SpriteScaleCalculatorNode:
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = (
        "pixel_width",
        "pixel_height",
    )
    FUNCTION = "calculate_pixels"
    CATEGORY = "image/transform"

    # Spirie is 5'0" (60") tall, rendered at 80px — the "width"/"height" widgets
    # below are already pixel targets, not inches, so no reference-derived
    # scale factor is applied to them (see web/spriteScaleCalculator.js for the
    # inches-per-pixel math used only for the feet/inches preview label).
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
        pixel_width = 0 if width <= 0 else max(1, width)
        pixel_height = 0 if height <= 0 else max(1, height)

        return (
            int(pixel_width),
            int(pixel_height),
        )
