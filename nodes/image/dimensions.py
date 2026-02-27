from __future__ import annotations


class PixelationDimensionsNode:
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_dimensions"
    CATEGORY = "image/transform"

    _PRESETS = {
        "Spirie": (979, 1562),
        "Custom": (0, 0),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls._PRESETS.keys()),),
            },
            "optional": {
                "custom_width": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 8192, "step": 1},
                ),
                "custom_height": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 8192, "step": 1},
                ),
            },
        }

    def get_dimensions(
        self, preset: str, custom_width: int = 1024, custom_height: int = 1024
    ):
        if preset == "Custom":
            width, height = custom_width, custom_height
        else:
            width, height = self._PRESETS[preset]
        return (width, height)
