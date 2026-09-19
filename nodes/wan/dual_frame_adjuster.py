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
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE", "INT", "INT")
    RETURN_NAMES = (
        "frames",
        "mask",
        "metadata",
        "raw_frames",
        "frame_count",
        "context_frames",
    )
    OUTPUT_TOOLTIPS = (
        "Only the context segment spanning every masked gap.",
        "Black for context frames and white for frames to inpaint.",
        "Metadata for WAN Unconnect Frames.",
        "Full internal WAN sequence passed through for WAN Unconnect Frames.",
        "Final full-animation frame count after WAN 1 + 4n adjustment.",
        "Actual number of frames in the context segment for the sampler length.",
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
                "context_frames": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 9998,
                        "step": 1,
                        "tooltip": "Number of unmasked context frames required before the first masked gap and after the last masked gap.",
                    },
                ),
                "preference": (
                    ["same frame count", "add frames"],
                    {
                        "default": "same frame count",
                        "tooltip": "Keep the requested core frame count when possible, or always round it up by adding frames.",
                    },
                ),
                "loop": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Build a seamless loop from section_1_frames alone. Its first frame becomes the end-cap anchor instead of core content, so it is not duplicated when the output repeats. section_2_frames, start_frame, and end_frame are ignored.",
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
        context_frames: int = 4,
        preference: str = "same frame count",
        loop: bool = False,
        section_2_frames: torch.Tensor | None = None,
        start_frame: torch.Tensor | None = None,
        end_frame: torch.Tensor | None = None,
    ):
        section_1_frames = _as_image_batch(section_1_frames, "section_1_frames")

        if loop:
            if int(section_1_frames.shape[0]) < 2:
                raise ValueError(
                    "loop requires section_1_frames to contain at least two frames: "
                    "one to close the loop against and at least one core frame."
                )
            # The first frame closes the loop rather than appearing as core
            # content; otherwise it is effectively duplicated when the output
            # repeats (it as core content, then again as the conditioning
            # target the last generated frame is pulled toward), producing a
            # visible pause. It is dropped from core and reused as the
            # end-cap anchor instead, which WANUnconnectFrames strips entirely.
            end_frame = section_1_frames[:1]
            section_1_frames = section_1_frames[1:]
            section_2_frames = None
            start_frame = None
        else:
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

        context_frames = int(context_frames)
        if context_frames < 0:
            raise ValueError("context_frames must be a non-negative integer.")

        if int(frames.shape[0]) != internal_target:
            raise RuntimeError(
                "Internal WAN frame calculation mismatch: "
                f"expected {internal_target}, got {frames.shape[0]}."
            )

        masked_frame_indices = torch.where(
            mask.reshape(mask.shape[0], -1).any(dim=1)
        )[0]
        if int(masked_frame_indices.numel()) == 0:
            raise ValueError(
                "WAN Connect Frames produced no masked gap to provide context for."
            )

        first_masked = int(masked_frame_indices[0])
        last_masked = int(masked_frame_indices[-1])
        if first_masked < context_frames:
            raise ValueError(
                f"context_frames={context_frames} requires that many unmasked frames "
                f"before the first masked gap; only {first_masked} are available."
            )
        trailing_context = internal_target - last_masked - 1
        if trailing_context < context_frames:
            raise ValueError(
                f"context_frames={context_frames} requires that many unmasked frames "
                f"after the last masked gap; only {trailing_context} are available."
            )

        segment_start = first_masked - context_frames
        segment_end = last_masked + 1 + context_frames
        segment_frames = frames[segment_start:segment_end]
        segment_mask = mask[segment_start:segment_end]
        base_segment_frame_count = int(segment_frames.shape[0])
        segment_padding_indices = []
        if (base_segment_frame_count - 1) % 4 != 0:
            if preference == "add frames":
                segment_padding_count = (1 - base_segment_frame_count) % 4
                insert_at = (last_masked - segment_start) + 1
                padding_frames = torch.ones(
                    (
                        segment_padding_count,
                        reference_shape[0],
                        reference_shape[1],
                        reference_shape[2],
                    ),
                    dtype=section_1_frames.dtype,
                    device=section_1_frames.device,
                )
                padding_mask = torch.ones(
                    (
                        segment_padding_count,
                        reference_shape[0],
                        reference_shape[1],
                    ),
                    dtype=torch.float32,
                    device=section_1_frames.device,
                )
                segment_frames = torch.cat(
                    (
                        segment_frames[:insert_at],
                        padding_frames,
                        segment_frames[insert_at:],
                    ),
                    dim=0,
                )
                segment_mask = torch.cat(
                    (
                        segment_mask[:insert_at],
                        padding_mask,
                        segment_mask[insert_at:],
                    ),
                    dim=0,
                )
                segment_padding_indices = list(
                    range(insert_at, insert_at + segment_padding_count)
                )
            elif preference == "same frame count":
                segment_context_count = (1 - base_segment_frame_count) % 4
                available_before = segment_start
                available_after = internal_target - segment_end
                try:
                    add_before, add_after = _balanced_split(
                        segment_context_count,
                        available_before,
                        available_after,
                    )
                except ValueError as error:
                    raise ValueError(
                        "WAN Connect Frames cannot add enough unmasked context frames "
                        "to make the sampler segment a valid 1 + 4n length."
                    ) from error

                prefix_frames = frames[segment_start - add_before:segment_start]
                suffix_frames = frames[segment_end:segment_end + add_after]
                prefix_mask = torch.zeros(
                    (add_before, reference_shape[0], reference_shape[1]),
                    dtype=torch.float32,
                    device=section_1_frames.device,
                )
                suffix_mask = torch.zeros(
                    (add_after, reference_shape[0], reference_shape[1]),
                    dtype=torch.float32,
                    device=section_1_frames.device,
                )
                segment_frames = torch.cat(
                    (prefix_frames, segment_frames, suffix_frames),
                    dim=0,
                )
                segment_mask = torch.cat(
                    (prefix_mask, segment_mask, suffix_mask),
                    dim=0,
                )
                segment_padding_indices = list(range(add_before))
                segment_padding_indices.extend(
                    range(
                        add_before + base_segment_frame_count,
                        add_before + base_segment_frame_count + add_after,
                    )
                )
            else:
                raise ValueError(
                    "WAN Connect Frames produced an invalid context segment length: "
                    f"{base_segment_frame_count}. WAN requires 1 + 4n frames; "
                    "use preference='add frames' or 'same frame count'."
                )

        segment_frame_count = int(segment_frames.shape[0])
        if (segment_frame_count - 1) % 4 != 0:
            raise RuntimeError(
                "WAN Connect Frames could not resolve the context segment to a valid 1 + 4n length."
            )

        metadata = json.dumps(
            {
                "format": "link-comfy-nodes/wan-connect-v2",
                "internal_frame_count": internal_target,
                "segment_frame_count": segment_frame_count,
                "base_segment_frame_count": base_segment_frame_count,
                "segment_padding_indices": segment_padding_indices,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "context_frames": context_frames,
                "masked_gap_count": int(masked_frame_indices.numel()),
                "remove_indices": remove_indices,
                "core_frame_count": core_target,
                "remove_first": start_frame is not None,
                "remove_last": end_frame is not None,
                "start_count": int(start_frame.shape[0]) if start_frame is not None else 0,
                "end_count": int(end_frame.shape[0]) if end_frame is not None else 0,
                "padding_count": padding_count,
            },
            separators=(",", ":"),
        )
        return (
            segment_frames,
            segment_mask,
            metadata,
            frames,
            internal_target,
            segment_frame_count,
        )


class WANUnconnectFrames:
    """Merge a sampled connection segment and remove WAN caps/padding."""

    CATEGORY = "conditioning/video_models"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    OUTPUT_TOOLTIPS = (
        "Core connection sequence with the inpainted masked gaps restored.",
    )
    FUNCTION = "unconnect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_frames": (
                    "IMAGE",
                    {
                        "tooltip": "The full internal sequence output by WAN Connect Frames.",
                    },
                ),
                "inpainted_frames": (
                    "IMAGE",
                    {
                        "tooltip": "The context segment output by WAN Connect Frames after inpainting.",
                    },
                ),
                "metadata": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect the metadata output from WAN Connect Frames.",
                    },
                ),
            },
        }

    @staticmethod
    def _parse_metadata(metadata_text: str) -> dict:
        try:
            metadata = json.loads(metadata_text)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("metadata is not valid WAN Connect Frames JSON.") from error
        if not isinstance(metadata, dict) or metadata.get("format") != "link-comfy-nodes/wan-connect-v2":
            raise ValueError("metadata is not recognized WAN Connect Frames metadata.")
        required = (
            "internal_frame_count",
            "segment_frame_count",
            "segment_start",
            "segment_end",
            "remove_indices",
            "core_frame_count",
        )
        for key in required:
            if key not in metadata:
                raise ValueError(f"metadata is missing {key}.")
        return metadata

    def unconnect(
        self,
        raw_frames: torch.Tensor,
        inpainted_frames: torch.Tensor,
        metadata: str,
    ):
        raw_frames = _as_image_batch(raw_frames, "raw_frames")
        inpainted_frames = _as_image_batch(inpainted_frames, "inpainted_frames")
        parsed = self._parse_metadata(metadata)

        internal_count = int(raw_frames.shape[0])
        expected_internal_count = int(parsed["internal_frame_count"])
        if internal_count != expected_internal_count:
            raise ValueError(
                "WAN Unconnect Frames received a different raw frame count than WAN Connect Frames output: "
                f"metadata expects {expected_internal_count}, received {internal_count}."
            )

        segment_count = int(inpainted_frames.shape[0])
        expected_segment_count = int(parsed["segment_frame_count"])
        if segment_count != expected_segment_count:
            raise ValueError(
                "WAN Unconnect Frames received a different inpainted segment length than WAN Connect Frames output: "
                f"metadata expects {expected_segment_count}, received {segment_count}."
            )
        if tuple(raw_frames.shape[1:]) != tuple(inpainted_frames.shape[1:]):
            raise ValueError(
                "raw_frames and inpainted_frames must have matching height, width, and channels."
            )

        segment_start = int(parsed["segment_start"])
        segment_end = int(parsed["segment_end"])
        if (
            segment_start < 0
            or segment_end <= segment_start
            or segment_end > internal_count
        ):
            raise ValueError("metadata contains invalid context segment bounds.")

        segment_padding_indices = parsed.get("segment_padding_indices", [])
        if not isinstance(segment_padding_indices, list):
            raise ValueError("metadata.segment_padding_indices must be a list.")
        segment_padding_indices = sorted(
            set(int(index) for index in segment_padding_indices)
        )
        if any(index < 0 or index >= segment_count for index in segment_padding_indices):
            raise ValueError("metadata contains an invalid context padding index.")

        segment_keep = torch.ones(
            segment_count,
            dtype=torch.bool,
            device=raw_frames.device,
        )
        if segment_padding_indices:
            segment_keep[
                torch.tensor(segment_padding_indices, device=raw_frames.device)
            ] = False
        merge_segment = inpainted_frames[segment_keep]

        expected_base_segment_count = int(
            parsed.get("base_segment_frame_count", segment_count - len(segment_padding_indices))
        )
        if int(merge_segment.shape[0]) != expected_base_segment_count:
            raise ValueError(
                "metadata context padding does not match the inpainted segment length."
            )
        if segment_end - segment_start != expected_base_segment_count:
            raise ValueError("metadata contains invalid context segment bounds.")

        merged = raw_frames.clone()
        merged[segment_start:segment_end] = merge_segment.to(
            device=raw_frames.device,
            dtype=raw_frames.dtype,
        )

        remove_indices = parsed["remove_indices"]
        if not isinstance(remove_indices, list):
            raise ValueError("metadata.remove_indices must be a list.")
        remove_indices = sorted(set(int(index) for index in remove_indices))
        if any(index < 0 or index >= internal_count for index in remove_indices):
            raise ValueError("metadata contains a frame index outside raw_frames.")

        keep = torch.ones(
            internal_count,
            dtype=torch.bool,
            device=raw_frames.device,
        )
        if remove_indices:
            keep[torch.tensor(remove_indices, device=raw_frames.device)] = False
        output_frames = merged[keep]

        expected_count = int(parsed["core_frame_count"])
        if int(output_frames.shape[0]) != expected_count:
            raise RuntimeError(
                "WAN Unconnect Frames produced an unexpected frame count: "
                f"expected {expected_count}, got {output_frames.shape[0]}."
            )
        if (expected_count - 1) % 4 != 0:
            raise RuntimeError(
                "WAN Unconnect Frames produced an invalid frame count: "
                f"{expected_count} is not 1 + 4n."
            )
        return (output_frames,)


# Compatibility aliases for external Python callers from earlier releases.
WANRemoveCapFrames = WANUnconnectFrames
WANBeginEndFrames = WANConnectFrames
