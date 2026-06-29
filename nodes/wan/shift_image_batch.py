from __future__ import annotations

import torch


class ShiftImageBatchNode:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("shifted_frames",)
    OUTPUT_TOOLTIPS = ("Circularly shifted image batch.",)
    FUNCTION = "shift"
    CATEGORY = "animation/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "amount": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Number of frames to shift.",
                    },
                ),
                "direction": (
                    ["left", "right"],
                    {
                        "default": "left",
                        "tooltip": "Direction to shift the batch.",
                    },
                ),
            }
        }

    def shift(self, frames: torch.Tensor, amount: int, direction: str):
        shift = -amount if direction == "left" else amount
        shifted = torch.roll(frames, shift, dims=0)
        return (shifted,)
