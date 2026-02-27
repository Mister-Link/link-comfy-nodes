from __future__ import annotations

import numpy as np


class WANFrameCalculatorNode:
    RETURN_TYPES: tuple[str, ...] = ("INT",)
    RETURN_NAMES: tuple[str, ...] = ("wan_frames",)
    FUNCTION: str = "calculate_wan_frames"
    CATEGORY: str = "animation/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10000,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "rounding_mode": (
                    ["nearest", "max", "min"],
                    {
                        "default": "nearest",
                    },
                ),
            }
        }

    def calculate_wan_frames(self, frame_count: int, rounding_mode: str):
        if frame_count <= 1:
            return (1,)

        if rounding_mode == "max":
            wan_frames = 1 + (int(np.ceil((frame_count - 1) / 4)) * 4)
        elif rounding_mode == "min":
            wan_frames = 1 + (int(np.floor((frame_count - 1) / 4)) * 4)
        else:
            wan_frames = 1 + (round((frame_count - 1) / 4) * 4)

        return (wan_frames,)
