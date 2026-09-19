"""Create and restore a WAN-valid midpoint loop segment."""

from __future__ import annotations

import json

import torch


LOOP_METADATA_FORMAT = "link-comfy-nodes/wan-loop-v2"


def _as_image_batch(frames: torch.Tensor, name: str = "frames") -> torch.Tensor:
    if not isinstance(frames, torch.Tensor):
        raise ValueError(f"{name} must be an IMAGE tensor.")
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError(
            f"{name} must have shape (frames, height, width, channels) "
            "with at least one frame."
        )
    return frames


def _valid_total_at_or_above(value: int) -> int:
    return value + ((1 - value) % 4)


def _valid_total_at_or_below(value: int) -> int:
    return value - ((value - 1) % 4)


class WANLoopFrames:
    """Extract a small midpoint context segment for WAN loop inpainting."""

    CATEGORY = "conditioning/video_models"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "INT", "IMAGE")
    RETURN_NAMES = (
        "frames",
        "mask",
        "frame_count",
        "metadata",
        "context_frames",
        "raw_frames",
    )
    OUTPUT_TOOLTIPS = (
        "Only the requested context segment: before-context, blank frames, after-context.",
        "Black for context frames and white for the blank frames to inpaint.",
        "Final full-animation frame count after WAN 1 + 4n adjustment.",
        "Metadata for WAN Unloop Frames to merge the inpainted segment back into the full animation.",
        "Actual number of frames in the context segment for the sampler length.",
        "Untouched original frame batch for WAN Unloop Frames.",
    )
    FUNCTION = "create"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "IMAGE",
                    {
                        "tooltip": "Full source animation. Only a midpoint context segment is output for sampling.",
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
                "context_frames": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 9998,
                        "step": 1,
                        "tooltip": "Number of source context frames included before and after the blank inpaint frames.",
                    },
                ),
                "preference": (
                    ["add frames", "cut frames"],
                    {
                        "default": "add frames",
                        "tooltip": "If the requested full-animation result is not 1 + 4n, round the blank-frame count up or down.",
                    },
                ),
            }
        }

    def create(
        self,
        frames: torch.Tensor,
        frames_to_cut: int = 0,
        frames_to_add: int = 0,
        context_frames: int = 4,
        preference: str = "add frames",
    ):
        frames = _as_image_batch(frames)
        frame_count = int(frames.shape[0])
        frames_to_cut = int(frames_to_cut)
        frames_to_add = int(frames_to_add)
        context_frames = int(context_frames)

        if frames_to_cut < 0:
            raise ValueError("frames_to_cut must be a non-negative integer.")
        if frames_to_add < 0:
            raise ValueError("frames_to_add must be non-negative.")
        if context_frames < 0:
            raise ValueError("context_frames must be non-negative.")
        if frames_to_cut >= frame_count:
            raise ValueError(
                "frames_to_cut must leave at least one source frame; "
                f"received {frames_to_cut} for {frame_count} input frames."
            )
        if preference not in {"add frames", "cut frames"}:
            raise ValueError("preference must be 'add frames' or 'cut frames'.")

        source_frame_count = frame_count - frames_to_cut
        requested_total = source_frame_count + frames_to_add

        if preference == "add frames":
            target_total = _valid_total_at_or_above(requested_total)
        else:
            target_total = _valid_total_at_or_below(requested_total)
            if target_total < source_frame_count:
                raise ValueError(
                    "cut frames preference cannot reach a valid WAN count without "
                    "removing more source frames or requesting more additions."
                )

        effective_frames_to_add = target_total - source_frame_count

        # Trim alternately from the beginning and end.
        left = 0
        right = frame_count
        for cut_index in range(frames_to_cut):
            if cut_index % 2 == 0:
                left += 1
            else:
                right -= 1

        trimmed = frames[left:right]

        # A fixed 50% scoot: play the latter half first, then the former half.
        midpoint = (source_frame_count + 1) // 2
        second_half = trimmed[midpoint:]
        first_half = trimmed[:midpoint]

        if context_frames > int(second_half.shape[0]) or context_frames > int(first_half.shape[0]):
            raise ValueError(
                f"context_frames={context_frames} is too large for the available "
                f"midpoint context ({second_half.shape[0]} before, {first_half.shape[0]} after)."
            )
        if context_frames == 0 and effective_frames_to_add == 0:
            raise ValueError(
                "The sampling segment would be empty; request context_frames or frames_to_add."
            )

        height, width, channels = (
            int(frames.shape[1]),
            int(frames.shape[2]),
            int(frames.shape[3]),
        )
        blank_frames = torch.ones(
            (effective_frames_to_add, height, width, channels),
            dtype=frames.dtype,
            device=frames.device,
        )

        before_context = second_half[-context_frames:] if context_frames else second_half[:0]
        after_context = first_half[:context_frames] if context_frames else first_half[:0]

        base_segment_count = (
            int(before_context.shape[0])
            + effective_frames_to_add
            + int(after_context.shape[0])
        )
        segment_padding_count = 0
        if (base_segment_count - 1) % 4 != 0:
            if preference == "add frames":
                segment_padding_count = (1 - base_segment_count) % 4
                padding_frames = torch.ones(
                    (segment_padding_count, height, width, channels),
                    dtype=frames.dtype,
                    device=frames.device,
                )
                blank_frames = torch.cat((blank_frames, padding_frames), dim=0)
            else:
                segment_cut_count = (base_segment_count - 1) % 4
                cut_after = min(segment_cut_count, int(after_context.shape[0]))
                if cut_after:
                    after_context = after_context[:-cut_after]
                    segment_cut_count -= cut_after
                if segment_cut_count:
                    if segment_cut_count > int(before_context.shape[0]):
                        raise ValueError(
                            "WAN Loop Frames cannot cut the context segment to a valid "
                            "1 + 4n length without removing the entire available context."
                        )
                    before_context = before_context[segment_cut_count:]
                if (
                    int(before_context.shape[0])
                    + effective_frames_to_add
                    + int(after_context.shape[0])
                ) < 1:
                    raise ValueError("WAN Loop Frames produced an empty context segment.")

        segment_frames = torch.cat(
            (before_context, blank_frames, after_context),
            dim=0,
        )

        segment_mask = torch.cat(
            (
                torch.zeros(
                    (int(before_context.shape[0]), height, width),
                    dtype=torch.float32,
                    device=frames.device,
                ),
                torch.ones(
                    (int(blank_frames.shape[0]), height, width),
                    dtype=torch.float32,
                    device=frames.device,
                ),
                torch.zeros(
                    (int(after_context.shape[0]), height, width),
                    dtype=torch.float32,
                    device=frames.device,
                ),
            ),
            dim=0,
        )

        segment_frame_count = int(segment_frames.shape[0])
        if segment_frame_count != (
            int(before_context.shape[0])
            + int(blank_frames.shape[0])
            + int(after_context.shape[0])
        ):
            raise RuntimeError("WAN Loop Frames context segment size mismatch.")
        if (segment_frame_count - 1) % 4 != 0:
            raise RuntimeError(
                "WAN Loop Frames could not resolve the context segment to a valid 1 + 4n length."
            )
        if (target_total - 1) % 4 != 0:
            raise RuntimeError(
                f"WAN Loop Frames produced invalid full frame count: {target_total}."
            )

        metadata = {
            "format": LOOP_METADATA_FORMAT,
            "input_frame_count": frame_count,
            "output_frame_count": target_total,
            "segment_frame_count": segment_frame_count,
            "frames_to_cut": frames_to_cut,
            "frames_to_add_requested": frames_to_add,
            "frames_to_add": effective_frames_to_add,
            "context_frames": context_frames,
            "trim_begin": left,
            "trim_end": frame_count - right,
            "source_frame_count": source_frame_count,
            "midpoint": midpoint,
            "segment_blank_offset": int(before_context.shape[0]),
            "segment_padding_count": segment_padding_count,
            "first_surviving_input_frame": left + 1,
        }

        return (
            segment_frames,
            segment_mask,
            target_total,
            json.dumps(metadata, separators=(",", ":")),
            segment_frame_count,
            frames,
        )


class WANUnloopFrames:
    """Merge an inpainted midpoint segment back into the raw animation."""

    CATEGORY = "conditioning/video_models"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    OUTPUT_TOOLTIPS = (
        "Raw animation with the inpainted midpoint frames inserted at the loop seam.",
    )
    FUNCTION = "restore"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_frames": (
                    "IMAGE",
                    {
                        "tooltip": "The untouched original animation batch passed to WAN Loop Frames.",
                    },
                ),
                "inpainted_frames": (
                    "IMAGE",
                    {
                        "tooltip": "The complete context segment output by WAN Loop Frames after inpainting.",
                    },
                ),
                "metadata": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect the metadata output from WAN Loop Frames.",
                    },
                ),
            }
        }

    @staticmethod
    def _parse_metadata(metadata_text: str) -> dict:
        try:
            metadata = json.loads(metadata_text)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("metadata is not valid WAN Loop Frames JSON.") from error
        if not isinstance(metadata, dict) or metadata.get("format") != LOOP_METADATA_FORMAT:
            raise ValueError("metadata is not recognized WAN Loop Frames metadata.")
        required = (
            "input_frame_count",
            "output_frame_count",
            "segment_frame_count",
            "frames_to_add",
            "segment_blank_offset",
            "trim_begin",
            "trim_end",
        )
        for key in required:
            if key not in metadata:
                raise ValueError(f"metadata is missing {key}.")
        return metadata

    def restore(
        self,
        raw_frames: torch.Tensor,
        inpainted_frames: torch.Tensor,
        metadata: str,
    ):
        raw_frames = _as_image_batch(raw_frames, "raw_frames")
        inpainted_frames = _as_image_batch(inpainted_frames, "inpainted_frames")
        metadata = self._parse_metadata(metadata)

        raw_count = int(raw_frames.shape[0])
        segment_count = int(inpainted_frames.shape[0])
        expected_raw_count = int(metadata["input_frame_count"])
        expected_segment_count = int(metadata["segment_frame_count"])

        if raw_count != expected_raw_count:
            raise ValueError(
                "WAN Unloop Frames requires raw_frames to be the untouched full batch: "
                f"metadata expects {expected_raw_count}, received {raw_count}. "
                f"inpainted_frames received {segment_count}; expected {expected_segment_count}."
            )

        if segment_count != expected_segment_count:
            raise ValueError(
                "WAN Unloop Frames requires inpainted_frames to be the sampled context segment: "
                f"metadata expects {expected_segment_count}, received {segment_count}. "
                f"raw_frames received {raw_count}; expected {expected_raw_count}."
            )

        if tuple(inpainted_frames.shape[1:]) != tuple(raw_frames.shape[1:]):
            raise ValueError(
                "raw_frames and inpainted_frames must have matching height, width, and channels."
            )

        trim_begin = int(metadata["trim_begin"])
        trim_end = int(metadata["trim_end"])
        if trim_begin < 0 or trim_end < 0 or trim_begin + trim_end >= raw_count:
            raise ValueError("metadata contains invalid trim boundaries.")

        source_frames = raw_frames[trim_begin:raw_count - trim_end]
        frames_to_add = int(metadata["frames_to_add"])
        blank_offset = int(metadata["segment_blank_offset"])
        inpainted_frames_to_merge = inpainted_frames[
            blank_offset:blank_offset + frames_to_add
        ]
        if int(inpainted_frames_to_merge.shape[0]) != frames_to_add:
            raise ValueError("metadata points outside the inpainted context segment.")

        restored_frames = torch.cat(
            (
                source_frames,
                inpainted_frames_to_merge.to(
                    device=source_frames.device,
                    dtype=source_frames.dtype,
                ),
            ),
            dim=0,
        )
        expected_output_count = int(metadata["output_frame_count"])
        if int(restored_frames.shape[0]) != expected_output_count:
            raise RuntimeError(
                "WAN Unloop Frames produced an unexpected frame count: "
                f"expected {expected_output_count}, got {restored_frames.shape[0]}."
            )

        return (restored_frames,)
