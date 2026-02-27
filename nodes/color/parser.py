from __future__ import annotations

from ...utils import parse_color_value


class ColorParserNode:
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
