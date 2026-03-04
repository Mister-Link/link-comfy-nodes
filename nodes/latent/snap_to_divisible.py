from __future__ import annotations


class SnapToDivisible:
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "snap"
    CATEGORY = "utils/math"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 1024, "min": 0, "max": 65536, "step": 1}),
                "divisible_by": ("INT", {"default": 16, "min": 1, "max": 65536, "step": 1}),
            }
        }

    def snap(self, value: int, divisible_by: int) -> tuple[int]:
        remainder = value % divisible_by
        if remainder == 0:
            return (value,)
        # Round to nearest: down if remainder <= half, up otherwise
        if remainder <= divisible_by / 2:
            return (value - remainder,)
        return (value + (divisible_by - remainder),)
