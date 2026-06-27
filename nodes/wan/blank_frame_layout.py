from __future__ import annotations

import torch


class LoopSCAILPoseFramesNode:
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = (
        "looped_frames",
        "total_frames",
        "overlap_frames",
    )
    OUTPUT_TOOLTIPS = (
        "Looped frame sequence ready for WAN SCAIL: [end portion] + [blank frames] + [start portion], "
        "with optional overlap frames prepended/appended. Feed this into SCAIL as the pose video.",
        "Actual total frame count of looped_frames. Equals total_frames input +4 when frame_overlap is on (2 prefix + 2 suffix).",
        "Number of overlap frames added (0 when frame_overlap is off, 4 when on).",
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
                        "tooltip": "Total WAN frame count — must be 1+n*4 (1, 5, 9, 13, … 25, 29, 33, 37…). Errors if invalid.",
                    },
                ),
                "blank_frames": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Number of black frames placed in the middle of the loop. SCAIL inpaints these to create a smooth pose transition between the end and start of the original sequence.",
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

    def calculate(self, frames: torch.Tensor, total_frames: int, blank_frames: int, frame_overlap: bool):
        if (total_frames - 1) % 4 != 0:
            nearest_low = 1 + ((total_frames - 1) // 4) * 4
            nearest_high = nearest_low + 4
            raise ValueError(
                f"total_frames={total_frames} is not a valid WAN frame count (must be 1+n*4). "
                f"Use {nearest_low} or {nearest_high}."
            )

        blank_frames = min(blank_frames, total_frames)
        remaining = total_frames - blank_frames

        start_num_frames = remaining // 2
        end_num_frames = remaining - start_num_frames

        _, H, W, C = frames.shape

        start_portion = frames[:start_num_frames] if start_num_frames > 0 else frames[:0]
        end_portion = frames[start_num_frames:start_num_frames + end_num_frames] if end_num_frames > 0 else frames[:0]
        black = torch.zeros(blank_frames, H, W, C, dtype=frames.dtype, device=frames.device)

        if frame_overlap:
            prefix = frames[start_num_frames - 2 : start_num_frames]
            suffix = frames[start_num_frames : start_num_frames + 2]
            looped_frames = torch.cat([prefix, end_portion, black, start_portion, suffix], dim=0)
        else:
            looped_frames = torch.cat([end_portion, black, start_portion], dim=0)

        out_total = looped_frames.shape[0]
        overlap_frames = 4 if frame_overlap else 0

        return (looped_frames, out_total, overlap_frames)
