from __future__ import annotations

import torch


class LoopSCAILPoseFramesNode:
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = (
        "frames",
        "total_frames",
        "overlap_frames",
        "shift",
    )
    OUTPUT_TOOLTIPS = (
        "Looped frame sequence ready for WAN SCAIL: [end portion] + [blank frames] + [start portion], "
        "with optional overlap frames prepended/appended. Feed this into SCAIL as the pose video.",
        "Actual total frame count of frames after trimming the source frames to keep the output WAN-valid.",
        "Number of overlap frames added (0 when frame_overlap is off, 4 when on).",
        "Amount to rotate the (overlap-trimmed) sequence left by to put original frame 1 back at position 1: "
        "end_portion length + blank_frames.",
    )
    FUNCTION = "calculate"
    CATEGORY = "animation/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "blank_frames": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Number of black inpaint frames placed in the middle of the loop. Looping/inpainting only applies when this is greater than 1. The node only enforces that the resulting total frame count is WAN-valid: 1+n*4 (1, 5, 9, 13, ...).",
                    },
                ),
                "frame_overlap": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When enabled, prepends the 2 frames immediately before the split point and appends "
                            "the 2 frames immediately after it. These overlap frames give WAN smooth temporal "
                            "context at both seam edges, which can improve loop smoothness. "
                            "The total output will be total_frames + 4 when enabled."
                        ),
                    },
                ),
            }
        }

    def calculate(self, frames: torch.Tensor, blank_frames: int, frame_overlap: bool):
        def is_wan_valid(count: int) -> bool:
            return count >= 1 and (count - 1) % 4 == 0

        input_frame_count = int(frames.shape[0])

        if blank_frames <= 1:
            if not is_wan_valid(input_frame_count):
                raise ValueError(
                    "Could not construct a WAN-valid output frame count from the provided inputs."
                )
            return (frames, input_frame_count, 0, 0)

        overlap_frames = 4 if frame_overlap else 0

        # Shift left 50%: rotate so the true end/begin seam sits near the middle.
        split = input_frame_count // 2
        end_portion = frames[split:]
        start_portion = frames[:split]

        end_num_frames = end_portion.shape[0]
        start_num_frames = start_portion.shape[0]

        if frame_overlap and (end_num_frames < 2 or start_num_frames < 2):
            raise ValueError(
                "frame_overlap requires at least 2 retained frames on both sides of the blank region."
            )

        _, H, W, C = frames.shape
        black = torch.zeros(blank_frames, H, W, C, dtype=frames.dtype, device=frames.device)
        seq = torch.cat([end_portion, black, start_portion], dim=0)

        if frame_overlap:
            # segment 1 = end_portion (before blanks), segment 2 = start_portion (after blanks).
            prefix = start_portion[-2:]
            suffix = end_portion[:2]
            looped_frames = torch.cat([prefix, seq, suffix], dim=0)
        else:
            looped_frames = seq

        out_total = looped_frames.shape[0]
        shift = end_num_frames + blank_frames

        if not is_wan_valid(out_total):
            raise ValueError(
                "Could not construct a WAN-valid output frame count from the provided inputs."
            )

        return (looped_frames, out_total, overlap_frames, shift)
