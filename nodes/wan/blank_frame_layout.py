from __future__ import annotations

import torch


def _is_wan_valid(count: int) -> bool:
    return count >= 1 and (count - 1) % 4 == 0


def _unshift_rotation(work_count: int, value: int, is_loop: bool) -> int:
    """Resolve the rotation needed to restore the original first frame.

    The shift node emits the full rotation (the second half plus blanks), but
    it is natural to enter the blank-frame amount manually in the unshift
    node.  A real loop rotation is always at least half of the trimmed batch;
    smaller loop values are therefore unambiguously blank-frame counts.
    """
    value = max(0, int(value))
    if is_loop and value < work_count // 2:
        blank_frames = value
        source_count = work_count - blank_frames
        value = source_count - source_count // 2 + blank_frames
    return value % work_count


class ShiftPoseFramesNode:
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = (
        "frames",
        "total_frames",
        "shift",
        "blank_frames",
        "is_loop",
    )
    OUTPUT_TOOLTIPS = (
        "Frame sequence shifted so the original seam sits near the middle, with loop blanks and overlap added, when blank_frames is greater than 0. Passed through unchanged when blank_frames is 0.",
        "Actual total frame count of the shifted output sequence.",
        "Amount to rotate the overlap-trimmed sequence left by to restore original frame order. 0 when blank_frames is 0.",
        "Number of blank frames inserted into the shifted loop, for reference.",
        "Whether loop blanks and overlap were added. Feed this into Unshift Pose Frames.",
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
                        "tooltip": "Number of black inpaint frames placed in the middle of the shifted loop. When this is greater than 0, the node also adds 2 overlap frames at each end and only validates the final total frame count. Set to 0 to pass the frames through untouched, with no shift and no blanks.",
                    },
                ),
            }
        }

    def calculate(self, frames: torch.Tensor, blank_frames: int):
        input_frame_count = int(frames.shape[0])
        split = input_frame_count // 2
        end_portion = frames[split:]
        start_portion = frames[:split]
        end_num_frames = int(end_portion.shape[0])
        start_num_frames = int(start_portion.shape[0])
        is_loop = blank_frames > 0

        if is_loop and (end_num_frames < 2 or start_num_frames < 2):
            raise ValueError(
                "Shift Pose Frames with blank_frames > 0 requires at least 2 frames on both sides of the split."
            )

        if is_loop:
            _, height, width, channels = frames.shape
            black = torch.zeros(
                blank_frames,
                height,
                width,
                channels,
                dtype=frames.dtype,
                device=frames.device,
            )
            seq = torch.cat([end_portion, black, start_portion], dim=0)
            prefix = start_portion[-2:]
            suffix = end_portion[:2]
            shifted_frames = torch.cat([prefix, seq, suffix], dim=0)
            shift = end_num_frames + blank_frames
        else:
            shifted_frames = frames
            shift = 0

        out_total = int(shifted_frames.shape[0])
        if not _is_wan_valid(out_total):
            raise ValueError(
                f"Shift Pose Frames received {input_frame_count} input frames and "
                f"{blank_frames} blank frames, producing {out_total} total frames "
                "(including 4 overlap frames when looping). WAN requires a frame "
                "count of 1 + (4*n): 1, 5, 9, 13, ... Adjust the input or blank "
                "frame count."
            )

        return (shifted_frames, out_total, shift, blank_frames, is_loop)


class UnshiftPoseFramesNode:
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "total_frames")
    OUTPUT_TOOLTIPS = (
        "Original frame order restored from Shift Pose Frames output, with overlap removed when is_loop is true.",
        "Actual total frame count after unshifting.",
    )
    FUNCTION = "calculate"
    CATEGORY = "animation/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "shift": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Connect the shift output from Shift Pose Frames. In loop mode, a small manual value is also accepted as the blank-frame count.",
                    },
                ),
                "is_loop": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Loop flag emitted by Shift Pose Frames. A positive shift also enables loop-layout recovery if this flag was not carried through.",
                    },
                ),
            }
        }

    def calculate(
        self,
        frames: torch.Tensor,
        shift: int,
        is_loop: bool,
    ):
        work = frames
        shift = max(0, int(shift))

        # Shift Pose Frames emits shift=0 for a pass-through and a positive
        # value whenever it inserted the midpoint seam/overlap layout.  Use
        # that invariant as a compatibility fallback when the BOOLEAN output
        # was not connected or was saved incorrectly.  Without this, a small
        # manual blank-frame value such as 2 is treated as a literal rotation
        # of the full batch, leaving the seam (and frame 14/15 in typical
        # batches) at the front instead of restoring source frame 1.
        loop_layout = bool(is_loop) or shift > 0

        if loop_layout:
            if int(work.shape[0]) < 4:
                raise ValueError(
                    "Looped Shift Pose Frames input must include the 4 overlap frames."
                )
            work = work[2:-2]

        work_count = int(work.shape[0])
        if work_count == 0:
            raise ValueError("No frames remain after removing overlap frames.")

        shift_mod = _unshift_rotation(work_count, shift, loop_layout)
        if shift_mod:
            work = torch.cat([work[shift_mod:], work[:shift_mod]], dim=0)

        return (work, int(work.shape[0]))


LoopSCAILPoseFramesNode = ShiftPoseFramesNode
