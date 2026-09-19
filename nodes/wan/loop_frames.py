"""Create a WAN-valid loop-oriented frame sequence around the midpoint."""

from __future__ import annotations

import json

import torch


class WANLoopFrames:
    """Rotate a batch around its midpoint and insert masked blank frames."""

    CATEGORY = "conditioning/video_models"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("frames", "mask", "frame_count", "loop_metadata")
    OUTPUT_TOOLTIPS = (
        "WAN-valid sequence starting at the midpoint, with inserted blank frames at the loop seam.",
        "Black for source frames and white for inserted blank frames.",
        "Actual output frame count after WAN 1 + 4n adjustment.",
        "Metadata for WAN Unloop Frames to restore the original surviving-frame order.",
    )
    FUNCTION = "create"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "IMAGE",
                    {
                        "tooltip": "Input frame batch to scoot around its fixed 50% midpoint.",
                    },
                ),
                "frames_to_cut": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9998,
                        "step": 1,
                        "tooltip": "Source frames removed alternately from the beginning and end; at least one frame is always preserved.",
                    },
                ),
                "frames_to_add": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9998,
                        "step": 1,
                        "tooltip": "Requested number of blank masked frames inserted at the midpoint seam.",
                    },
                ),
                "preference": (
                    ["add frames", "cut frames"],
                    {
                        "default": "add frames",
                        "tooltip": "If the requested result is not 1 + 4n frames, round the blank-frame count up or down.",
                    },
                ),
            }
        }

    @staticmethod
    def _as_image_batch(frames: torch.Tensor) -> torch.Tensor:
        if not isinstance(frames, torch.Tensor):
            raise ValueError("frames must be an IMAGE tensor.")
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4:
            raise ValueError(
                "frames must have shape (frames, height, width, channels); "
                f"received {tuple(frames.shape)}."
            )
        if frames.shape[0] < 1:
            raise ValueError("frames must contain at least one frame.")
        return frames

    @staticmethod
    def _valid_total_at_or_above(value: int) -> int:
        return value + ((1 - value) % 4)

    @staticmethod
    def _valid_total_at_or_below(value: int) -> int:
        return value - ((value - 1) % 4)

    def create(
        self,
        frames: torch.Tensor,
        frames_to_cut: int = 0,
        frames_to_add: int = 0,
        preference: str = "add frames",
    ):
        frames = self._as_image_batch(frames)
        frame_count = int(frames.shape[0])
        frames_to_cut = int(frames_to_cut)
        frames_to_add = int(frames_to_add)

        if frames_to_cut < 0:
            raise ValueError("frames_to_cut must be a non-negative integer.")
        if frames_to_add < 0:
            raise ValueError("frames_to_add must be non-negative.")
        if frames_to_cut >= frame_count:
            raise ValueError(
                "frames_to_cut must leave at least one source frame; "
                f"received {frames_to_cut} for {frame_count} input frames."
            )
        if preference not in {"add frames", "cut frames"}:
            raise ValueError("preference must be 'add frames' or 'cut frames'.")

        source_frames = frame_count - frames_to_cut
        requested_total = source_frames + frames_to_add

        if preference == "add frames":
            target_total = self._valid_total_at_or_above(requested_total)
        else:
            target_total = self._valid_total_at_or_below(requested_total)
            if target_total < source_frames:
                raise ValueError(
                    "cut frames preference cannot reach a valid WAN count without "
                    "removing more source frames or requesting more additions."
                )

        effective_frames_to_add = target_total - source_frames
        # Trim alternately from the beginning and end. The strict
        # frames_to_cut < frame_count check above leaves at least one frame,
        # so neither side can consume the final remaining frame.
        left = 0
        right = frame_count
        for cut_index in range(frames_to_cut):
            if cut_index % 2 == 0:
                left += 1
            else:
                right -= 1
        trimmed = frames[left:right]

        # A fixed 50% scoot: play the latter half first, then the former half.
        # Keeping the extra frame in the first half makes the split stable for
        # odd source counts and matches the workflow's ceil(N / 2) midpoint.
        midpoint = (source_frames + 1) // 2
        second_half = trimmed[midpoint:]
        first_half = trimmed[:midpoint]

        height, width = int(frames.shape[1]), int(frames.shape[2])
        blank_frames = torch.ones(
            (effective_frames_to_add, height, width, int(frames.shape[3])),
            dtype=frames.dtype,
            device=frames.device,
        )
        source_mask = torch.zeros(
            (source_frames, height, width),
            dtype=torch.float32,
            device=frames.device,
        )
        blank_mask = torch.ones(
            (effective_frames_to_add, height, width),
            dtype=torch.float32,
            device=frames.device,
        )

        output_frames = torch.cat((second_half, blank_frames, first_half), dim=0)
        output_mask = torch.cat((source_mask[midpoint:], blank_mask, source_mask[:midpoint]), dim=0)

        if int(output_frames.shape[0]) != target_total or (target_total - 1) % 4 != 0:
            raise RuntimeError(
                "WAN Loop Frames produced an invalid frame count: "
                f"{output_frames.shape[0]}."
            )

        loop_metadata = {
            "format": "link-comfy-nodes/wan-loop-v1",
            "input_frame_count": frame_count,
            "output_frame_count": target_total,
            "frames_to_cut": frames_to_cut,
            "frames_to_add_requested": frames_to_add,
            "frames_to_add": effective_frames_to_add,
            "trim_begin": left,
            "trim_end": frame_count - right,
            "source_frame_count": source_frames,
            "midpoint": midpoint,
            # WAN Unloop Frames rotates this many positions left to put the
            # first surviving source frame back at output index zero.
            "undo_shift": int(second_half.shape[0]) + effective_frames_to_add,
            "first_surviving_input_frame": left + 1,
        }
        return (
            output_frames,
            output_mask,
            target_total,
            json.dumps(loop_metadata, separators=(",", ":")),
        )


class WANUnloopFrames:
    """Restore the surviving source-frame order from WAN Loop Frames output."""

    CATEGORY = "conditioning/video_models"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "frame_count")
    OUTPUT_TOOLTIPS = (
        "Frames rotated back so the first surviving source frame is first.",
        "Actual output frame count after restoring order.",
    )
    FUNCTION = "restore"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "loop_metadata": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect the loop_metadata output from WAN Loop Frames.",
                    },
                ),
            }
        }

    @staticmethod
    def _as_image_batch(frames: torch.Tensor) -> torch.Tensor:
        if not isinstance(frames, torch.Tensor):
            raise ValueError("frames must be an IMAGE tensor.")
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4 or frames.shape[0] < 1:
            raise ValueError(
                "frames must have shape (frames, height, width, channels) "
                "with at least one frame."
            )
        return frames

    @staticmethod
    def _parse_metadata(loop_metadata: str) -> dict:
        try:
            metadata = json.loads(loop_metadata)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("loop_metadata is not valid WAN Loop Frames JSON.") from error
        if not isinstance(metadata, dict) or metadata.get("format") != "link-comfy-nodes/wan-loop-v1":
            raise ValueError("loop_metadata is not recognized WAN Loop Frames metadata.")
        for key in ("output_frame_count", "undo_shift"):
            if key not in metadata:
                raise ValueError(f"loop_metadata is missing {key}.")
        return metadata

    def restore(
        self,
        frames: torch.Tensor,
        loop_metadata: str,
    ):
        frames = self._as_image_batch(frames)
        metadata = self._parse_metadata(loop_metadata)
        frame_count = int(frames.shape[0])
        expected_count = int(metadata["output_frame_count"])
        if frame_count != expected_count:
            raise ValueError(
                "WAN Unloop Frames received a different frame count than WAN Loop Frames produced: "
                f"metadata expects {expected_count}, received {frame_count}."
            )

        shift = int(metadata["undo_shift"]) % frame_count
        if shift:
            restored_frames = torch.cat((frames[shift:], frames[:shift]), dim=0)
        else:
            restored_frames = frames

        return restored_frames, frame_count
