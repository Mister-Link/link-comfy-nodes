from __future__ import annotations

import torch


class LoopSCAILPoseFramesNode:
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "looped_frames",
        "start_num_frames",
        "num_blank_frames",
        "end_num_frames",
        "total_frames",
    )
    FUNCTION = "calculate"
    CATEGORY = "animation/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "total_frames": (
                    "INT",
                    {
                        "default": 37,
                        "min": 1,
                        "max": 10000,
                        "step": 4,
                        "tooltip": "Total WAN frame count (1+n*4). Snapped automatically if not valid.",
                    },
                ),
                "blank_frames": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Black frames in the middle — SCAIL inpaints loop-seam poses here.",
                    },
                ),
            }
        }

    def calculate(self, frames: torch.Tensor, total_frames: int, blank_frames: int):
        # Snap to nearest valid WAN count: 1 + (n * 4)
        total_frames = 1 + round((total_frames - 1) / 4) * 4
        total_frames = max(1, total_frames)

        blank_frames = min(blank_frames, total_frames)
        remaining = total_frames - blank_frames

        start_num_frames = remaining // 2
        end_num_frames = remaining - start_num_frames

        _, H, W, C = frames.shape

        start_portion = frames[:start_num_frames] if start_num_frames > 0 else frames[:0]
        end_portion = frames[start_num_frames:start_num_frames + end_num_frames] if end_num_frames > 0 else frames[:0]
        black = torch.zeros(blank_frames, H, W, C, dtype=frames.dtype, device=frames.device)

        looped_frames = torch.cat([end_portion, black, start_portion], dim=0)

        return (looped_frames, start_num_frames, blank_frames, end_num_frames, total_frames)
