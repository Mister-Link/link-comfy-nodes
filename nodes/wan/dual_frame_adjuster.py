"""Build and optionally de-cap a WAN-compatible connection sequence."""

from __future__ import annotations

import json

import torch


def _next_wan_frame_count(value: int) -> int:
    value = max(1, int(value))
    return value + ((1 - value) % 4)


def _previous_wan_frame_count(value: int) -> int:
    value = max(1, int(value))
    return value - ((value - 1) % 4)


def _as_image_batch(frames: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(frames, torch.Tensor):
        raise ValueError(f"{name} must be an IMAGE tensor.")

    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4:
        raise ValueError(
            f"{name} must have shape (frames, height, width, channels); "
            f"received {tuple(frames.shape)}."
        )
    if frames.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one frame.")
    return frames


def _as_mask_batch(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask must be a MASK tensor.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(
            "mask must have shape (frames, height, width); "
            f"received {tuple(mask.shape)}."
        )
    return mask


def _allocate_evenly(amount: int, gap_count: int) -> list[int]:
    """Allocate an integer amount as evenly as possible across gaps."""
    amount = max(0, int(amount))
    gap_count = max(0, int(gap_count))
    if gap_count == 0:
        return []

    base, remainder = divmod(amount, gap_count)
    return [base + (1 if index < remainder else 0) for index in range(gap_count)]


def _balanced_split(amount: int, first_capacity: int, second_capacity: int) -> tuple[int, int]:
    """Split amount as evenly as possible within both capacities."""
    amount = max(0, int(amount))
    first_capacity = max(0, int(first_capacity))
    second_capacity = max(0, int(second_capacity))

    min_first = max(0, amount - second_capacity)
    max_first = min(first_capacity, amount)
    if min_first > max_first:
        raise ValueError("The requested split exceeds the available capacity.")

    balanced_first = min(max_first, max(min_first, amount // 2))
    balanced_first_ceil = min(max_first, max(min_first, (amount + 1) // 2))
    candidates = {min_first, max_first, balanced_first, balanced_first_ceil}
    first = min(
        candidates,
        key=lambda value: (abs((2 * value) - amount), -value),
    )
    return first, amount - first


class WANConnectFrames:
    """Create a WAN sequence with optional boundary caps and cleanup metadata."""

    CATEGORY = "conditioning/video_models"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("frames", "mask", "cap_info")
    OUTPUT_TOOLTIPS = (
        "Optional start, section 1, white connection frames, section 2, and optional end frames.",
        "One mask per frame: black for supplied images and white for connection frames.",
        "Metadata for WAN Remove Cap Frames; connect this output to its cap_info input.",
    )
    FUNCTION = "create"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_1_frames": (
                    "IMAGE",
                    {
                        "tooltip": "Required first sequence. Supply only the frames you want to keep.",
                    },
                ),
                "transition_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Number of white frames to use for transitions between sections. Same-frame-count mode may trim source frames to preserve the requested total.",
                    },
                ),
                "preference": (
                    ["same frame count", "add frames"],
                    {
                        "default": "same frame count",
                        "tooltip": "Keep the requested core frame count when possible, or always round it up by adding frames.",
                    },
                ),
            },
            "optional": {
                "section_2_frames": (
                    "IMAGE",
                    {
                        "tooltip": "Optional second sequence. Either provide this or end_frame.",
                    },
                ),
                "start_frame": (
                    "IMAGE",
                    {
                        "tooltip": "Optional leading cap image. It is repeated to fill its share of four removable WAN cap frames.",
                    },
                ),
                "end_frame": (
                    "IMAGE",
                    {
                        "tooltip": "Optional trailing cap image. It is repeated to fill its share of four removable WAN cap frames.",
                    },
                ),
            },
        }

    def create(
        self,
        section_1_frames: torch.Tensor,
        transition_frames: int = 0,
        preference: str = "same frame count",
        section_2_frames: torch.Tensor | None = None,
        start_frame: torch.Tensor | None = None,
        end_frame: torch.Tensor | None = None,
    ):
        section_1_frames = _as_image_batch(section_1_frames, "section_1_frames")
        section_2_frames = (
            _as_image_batch(section_2_frames, "section_2_frames")
            if section_2_frames is not None
            else None
        )
        start_frame = (
            _as_image_batch(start_frame, "start_frame")
            if start_frame is not None
            else None
        )
        end_frame = (
            _as_image_batch(end_frame, "end_frame")
            if end_frame is not None
            else None
        )

        if section_2_frames is None and end_frame is None:
            raise ValueError(
                "At least one of section_2_frames or end_frame must be provided."
            )

        reference_shape = tuple(section_1_frames.shape[1:])
        for name, frames in (
            ("section_2_frames", section_2_frames),
            ("start_frame", start_frame),
            ("end_frame", end_frame),
        ):
            if frames is not None and tuple(frames.shape[1:]) != reference_shape:
                raise ValueError(
                    "All frame inputs must have the same height, width, and channel count; "
                    f"{name} has shape {tuple(frames.shape[1:])}, expected {reference_shape}."
                )

        def on_output_device(frames: torch.Tensor | None) -> torch.Tensor | None:
            if frames is None:
                return None
            return frames.to(
                device=section_1_frames.device,
                dtype=section_1_frames.dtype,
            )

        section_2_frames = on_output_device(section_2_frames)
        start_frame = on_output_device(start_frame)
        end_frame = on_output_device(end_frame)

        # WAN requires the complete sequence to have a length of 1 + 4*n.
        # Reserve exactly four removable cap frames, splitting them evenly
        # between the supplied boundaries. A cap input represents one boundary
        # image; repeat that image when its boundary receives multiple slots.
        cap_boundary_count = int(start_frame is not None) + int(end_frame is not None)
        cap_allocation = _allocate_evenly(4, cap_boundary_count)
        cap_allocation_index = 0
        if start_frame is not None:
            start_frame = start_frame[:1].repeat(
                (cap_allocation[cap_allocation_index], 1, 1, 1)
            )
            cap_allocation_index += 1
        if end_frame is not None:
            end_frame = end_frame[-1:].repeat(
                (cap_allocation[cap_allocation_index], 1, 1, 1)
            )

        core_source_sequences = [section_1_frames]
        if section_2_frames is not None:
            core_source_sequences.append(section_2_frames)
        core_source_count = sum(
            int(sequence.shape[0]) for sequence in core_source_sequences
        )
        transition_frames = max(0, int(transition_frames))
        requested_core_count = core_source_count + transition_frames
        minimum_core_count = len(core_source_sequences)
        add_target = _next_wan_frame_count(
            max(minimum_core_count, requested_core_count)
        )
        same_count_add_target = _next_wan_frame_count(
            max(minimum_core_count, core_source_count)
        )
        same_count_remove_target = _previous_wan_frame_count(core_source_count)
        can_remove_to_same_count = same_count_remove_target >= minimum_core_count

        if preference == "add frames":
            core_target = add_target
        elif can_remove_to_same_count and core_source_count - same_count_remove_target < same_count_add_target - core_source_count:
            core_target = same_count_remove_target
        else:
            core_target = same_count_add_target

        section_1_capacity = max(0, int(section_1_frames.shape[0]) - 1)
        section_2_capacity = (
            max(0, int(section_2_frames.shape[0]) - 1)
            if section_2_frames is not None
            else 0
        )
        source_cuts = min(
            max(0, core_source_count + transition_frames - core_target),
            section_1_capacity + section_2_capacity,
        )
        section_1_cut, section_2_cut = _balanced_split(
            source_cuts,
            section_1_capacity,
            section_2_capacity,
        )
        section_1_output = section_1_frames[: -section_1_cut or None]
        section_2_output = (
            section_2_frames[section_2_cut:]
            if section_2_frames is not None
            else None
        )
        remaining_core_source_count = (
            int(section_1_output.shape[0])
            + (int(section_2_output.shape[0]) if section_2_output is not None else 0)
        )
        core_blank_count = max(0, core_target - remaining_core_source_count)

        output_sequences: list[tuple[str, torch.Tensor]] = []
        if start_frame is not None:
            output_sequences.append(("start_cap", start_frame))
        output_sequences.append(("section_1", section_1_output))
        if section_2_output is not None:
            output_sequences.append(("section_2", section_2_output))
        if end_frame is not None:
            output_sequences.append(("end_cap", end_frame))

        cap_count = sum(
            int(sequence.shape[0])
            for name, sequence in output_sequences
            if name.endswith("_cap")
        )
        internal_target = _next_wan_frame_count(core_target + cap_count)
        blank_count = internal_target - remaining_core_source_count - cap_count
        blank_allocations = _allocate_evenly(blank_count, len(output_sequences) - 1)

        frame_parts = []
        mask_parts = []
        cap_indices = []
        blank_indices_by_gap = []
        frame_cursor = 0
        for index, (name, sequence) in enumerate(output_sequences):
            frame_parts.append(sequence)
            sequence_count = int(sequence.shape[0])
            mask_parts.append(
                torch.zeros(
                    (sequence_count, reference_shape[0], reference_shape[1]),
                    dtype=torch.float32,
                    device=section_1_frames.device,
                )
            )
            if name.endswith("_cap"):
                cap_indices.extend(range(frame_cursor, frame_cursor + sequence_count))
            frame_cursor += sequence_count

            if index >= len(blank_allocations):
                continue
            gap_count = blank_allocations[index]
            blank_indices_by_gap.append(
                list(range(frame_cursor, frame_cursor + gap_count))
            )
            frame_parts.append(
                torch.ones(
                    (gap_count, *reference_shape),
                    dtype=section_1_frames.dtype,
                    device=section_1_frames.device,
                )
            )
            mask_parts.append(
                torch.ones(
                    (gap_count, reference_shape[0], reference_shape[1]),
                    dtype=torch.float32,
                    device=section_1_frames.device,
                )
            )
            frame_cursor += gap_count

        frames = torch.cat(frame_parts, dim=0)
        mask = torch.cat(mask_parts, dim=0)

        padding_count = internal_target - cap_count - core_target
        preferred_padding_indices = []
        if start_frame is not None and blank_indices_by_gap:
            preferred_padding_indices.extend(blank_indices_by_gap[0])
        if end_frame is not None and blank_indices_by_gap:
            preferred_padding_indices.extend(blank_indices_by_gap[-1])
        for gap_indices in blank_indices_by_gap:
            preferred_padding_indices.extend(gap_indices)
        preferred_padding_indices = list(dict.fromkeys(preferred_padding_indices))
        padding_indices = preferred_padding_indices[:padding_count]
        remove_indices = sorted(set(cap_indices + padding_indices))

        cap_info = json.dumps(
            {
                "version": 1,
                "remove_first": start_frame is not None,
                "remove_last": end_frame is not None,
                "start_count": int(start_frame.shape[0]) if start_frame is not None else 0,
                "end_count": int(end_frame.shape[0]) if end_frame is not None else 0,
                "padding_count": padding_count,
                "remove_indices": remove_indices,
                "core_frame_count": core_target,
                "internal_frame_count": internal_target,
            },
            separators=(",", ":"),
        )

        if int(frames.shape[0]) != internal_target:
            raise RuntimeError(
                "Internal WAN frame calculation mismatch: "
                f"expected {internal_target}, got {int(frames.shape[0])}."
            )
        return (frames, mask, cap_info)


class WANRemoveCapFrames:
    """Remove WAN Connect Frames boundary caps and their padding metadata."""

    CATEGORY = "conditioning/video_models"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("frames", "mask")
    FUNCTION = "remove"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "cap_info": (
                    "STRING",
                    {
                        "tooltip": "Connect the cap_info output from WAN Connect Frames.",
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Optional mask to trim along with the removed cap frames.",
                    },
                ),
            },
        }

    def remove(
        self,
        frames: torch.Tensor,
        cap_info: str,
        mask: torch.Tensor | None = None,
    ):
        frames = _as_image_batch(frames, "frames")
        if mask is not None:
            mask = _as_mask_batch(mask)
            if int(mask.shape[0]) != int(frames.shape[0]):
                raise ValueError(
                    "frames and mask must contain the same number of frames; "
                    f"received {frames.shape[0]} and {mask.shape[0]}."
                )
            if tuple(mask.shape[1:]) != tuple(frames.shape[1:3]):
                raise ValueError(
                    "frames and mask must have matching height and width; "
                    f"received {tuple(frames.shape[1:3])} and {tuple(mask.shape[1:])}."
                )

        try:
            metadata = json.loads(cap_info)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("cap_info is not valid WAN Connect Frames metadata.") from error
        if not isinstance(metadata, dict) or metadata.get("version") != 1:
            raise ValueError("cap_info is not recognized WAN Connect Frames metadata.")

        remove_indices = metadata.get("remove_indices", [])
        if not isinstance(remove_indices, list):
            raise ValueError("cap_info.remove_indices must be a list.")
        remove_indices = sorted(set(int(index) for index in remove_indices))
        if any(index < 0 or index >= int(frames.shape[0]) for index in remove_indices):
            raise ValueError("cap_info contains a frame index outside the input batch.")

        keep = torch.ones(
            int(frames.shape[0]),
            dtype=torch.bool,
            device=frames.device,
        )
        if remove_indices:
            keep[torch.tensor(remove_indices, device=frames.device)] = False
        output_frames = frames[keep]
        if mask is None:
            output_mask = torch.zeros(
                (int(output_frames.shape[0]), *frames.shape[1:3]),
                dtype=torch.float32,
                device=frames.device,
            )
        else:
            output_mask = mask.to(device=frames.device)[keep]

        expected_count = metadata.get("core_frame_count")
        if expected_count is not None and int(output_frames.shape[0]) != int(expected_count):
            raise RuntimeError(
                "WAN cap removal produced an unexpected frame count: "
                f"expected {expected_count}, got {output_frames.shape[0]}."
            )
        if (int(output_frames.shape[0]) - 1) % 4 != 0:
            raise RuntimeError(
                "WAN cap removal produced an invalid frame count: "
                f"{output_frames.shape[0]} is not 1 + 4n."
            )
        return (output_frames, output_mask)


# Keep imports from the previous node implementation working for external code.
WANBeginEndFrames = WANConnectFrames
