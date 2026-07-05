from __future__ import annotations

import torch


class LoopSCAILPoseFramesNode:
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = (
        "frames",
        "total_frames",
        "overlap_frames",
        "insertion_point",
    )
    OUTPUT_TOOLTIPS = (
        "Looped frame sequence ready for WAN SCAIL: [end portion] + [blank frames] + [start portion], "
        "with optional overlap frames prepended/appended. Feed this into SCAIL as the pose video.",
        "Actual total frame count of frames after trimming the source frames to keep the output WAN-valid.",
        "Number of overlap frames added (0 when frame_overlap is off, 4 when on).",
        "0-based index of the first frame after the blank frames in frames.",
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
                        "step": 1,
                        "tooltip": "Maximum target frame count before optional overlap. The node may trim source frames down from this target to keep blank frames and output frames WAN-valid.",
                    },
                ),
                "blank_frames": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Number of black inpaint frames placed in the middle of the loop. Must be 0 or a WAN-valid count: 1+n*4 (1, 5, 9, 13, ...).",
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
        def is_wan_valid(count: int) -> bool:
            return count >= 1 and (count - 1) % 4 == 0

        def nearest_wan_values(count: int) -> tuple[int, int]:
            nearest_low = 1 + ((max(1, count) - 1) // 4) * 4
            nearest_high = nearest_low + 4
            return nearest_low, nearest_high

        input_frame_count = int(frames.shape[0])

        if blank_frames <= 0:
            if not is_wan_valid(input_frame_count):
                nearest_low, nearest_high = nearest_wan_values(input_frame_count)
                raise ValueError(
                    f"frames input has {input_frame_count} frames. With blank_frames=0, "
                    f"the input must be WAN-valid (1+n*4). Use {nearest_low} or {nearest_high}."
                )
            return (frames, input_frame_count, 0, 0)

        if not is_wan_valid(blank_frames):
            nearest_low, nearest_high = nearest_wan_values(blank_frames)
            raise ValueError(
                f"blank_frames={blank_frames} is not a valid WAN frame count (must be 1+n*4 or 0). "
                f"Use {nearest_low} or {nearest_high}."
            )

        overlap_frames = 4 if frame_overlap else 0
        target_source_frames = max(0, min(input_frame_count, total_frames - blank_frames))
        source_frame_count = target_source_frames - (target_source_frames % 4)

        if frame_overlap and 0 < source_frame_count < 4:
            source_frame_count = 0

        if frame_overlap and source_frame_count == 0 and input_frame_count > 0:
            raise ValueError(
                "frame_overlap requires at least 4 usable source frames after trimming. "
                "Increase total_frames, reduce blank_frames, or disable frame_overlap."
            )

        start_num_frames = source_frame_count // 2
        end_num_frames = source_frame_count - start_num_frames

        _, H, W, C = frames.shape

        start_portion = (
            frames[:start_num_frames] if start_num_frames > 0 else frames[:0]
        )
        end_portion = (
            frames[input_frame_count - end_num_frames : input_frame_count]
            if end_num_frames > 0
            else frames[:0]
        )
        black = torch.zeros(blank_frames, H, W, C, dtype=frames.dtype, device=frames.device)

        if frame_overlap:
            if start_num_frames < 2 or end_num_frames < 2:
                raise ValueError(
                    "frame_overlap requires at least 2 retained frames on both sides of the insertion point."
                )
            prefix = end_portion[-2:]
            suffix = start_portion[:2]
            looped_frames = torch.cat([prefix, end_portion, black, start_portion, suffix], dim=0)
        else:
            looped_frames = torch.cat([end_portion, black, start_portion], dim=0)

        out_total = looped_frames.shape[0]
        insertion_point = (2 if frame_overlap else 0) + end_num_frames + blank_frames

        if not is_wan_valid(out_total):
            raise ValueError(
                "Could not construct a WAN-valid output frame count from the provided inputs."
            )

        return (looped_frames, out_total, overlap_frames, insertion_point)
