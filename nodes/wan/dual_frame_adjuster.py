"""Build a WAN-compatible connection sequence from up to four frame sections."""

from __future__ import annotations

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


def _transition_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return a device-independent visual distance for two boundary frames."""
    distance = torch.mean(torch.abs(first.float() - second.float()))
    return float(distance.detach().cpu().item())


def _allocate_weighted(amount: int, weights: list[float]) -> list[int]:
    """Allocate an integer amount proportionally, using largest remainders."""
    amount = max(0, int(amount))
    if not weights:
        return []

    safe_weights = [max(0.0, float(weight)) for weight in weights]
    weight_total = sum(safe_weights)
    if weight_total <= 0.0:
        safe_weights = [1.0] * len(weights)
        weight_total = float(len(weights))

    exact = [amount * weight / weight_total for weight in safe_weights]
    allocation = [int(value) for value in exact]
    remainder = amount - sum(allocation)
    order = sorted(
        range(len(weights)),
        key=lambda index: (exact[index] - allocation[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        allocation[index] += 1
    return allocation


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
    """Create WAN frames and an inpaint mask from connected frame sections."""

    CATEGORY = "conditioning/video_models"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("frames", "mask")
    OUTPUT_TOOLTIPS = (
        "Optional start, section 1, white connection frames, section 2, and optional end frames.",
        "One mask per frame: black for supplied images and white for connection frames.",
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
                "frames_to_add_between": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": "Total white connection frames to distribute across the available transitions.",
                    },
                ),
                "preference": (
                    ["balanced", "add frames", "remove frames"],
                    {
                        "default": "balanced",
                        "tooltip": "When WAN rounding is needed, choose whether to add white frames, remove source frames, or choose the closer result.",
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
                        "tooltip": "Optional leading frame or batch. White connection frames are allocated between it and the next sequence when useful.",
                    },
                ),
                "end_frame": (
                    "IMAGE",
                    {
                        "tooltip": "Optional trailing frame or batch. Either provide this or section_2_frames.",
                    },
                ),
            },
        }

    def create(
        self,
        section_1_frames: torch.Tensor,
        frames_to_add_between: int = 0,
        preference: str = "balanced",
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

        sequences = []
        if start_frame is not None:
            sequences.append(start_frame)
        sequences.append(section_1_frames)
        if section_2_frames is not None:
            sequences.append(section_2_frames)
        if end_frame is not None:
            sequences.append(end_frame)

        input_total = sum(int(sequence.shape[0]) for sequence in sequences)
        frames_to_add_between = max(0, int(frames_to_add_between))
        requested_total = input_total + frames_to_add_between
        minimum_total = len(sequences)

        add_target = _next_wan_frame_count(max(minimum_total, requested_total))
        add_delta = add_target - requested_total
        remove_target = _previous_wan_frame_count(requested_total)
        can_remove_to_wan = remove_target >= minimum_total
        remove_delta = requested_total - remove_target if can_remove_to_wan else None

        if preference == "add frames":
            final_total = add_target
            extra_cuts = 0
            extra_blank_frames = add_delta
        elif preference == "remove frames" and can_remove_to_wan:
            final_total = remove_target
            extra_cuts = remove_delta
            extra_blank_frames = 0
        elif preference == "remove frames":
            final_total = add_target
            extra_cuts = 0
            extra_blank_frames = add_delta
        elif not can_remove_to_wan or add_delta <= remove_delta:
            final_total = add_target
            extra_cuts = 0
            extra_blank_frames = add_delta
        else:
            final_total = remove_target
            extra_cuts = remove_delta
            extra_blank_frames = 0

        # Only section 1's tail and section 2's head are auto-trimmable. The
        # optional boundary inputs are anchors and remain intact.
        section_1_capacity = max(0, int(section_1_frames.shape[0]) - 1)
        section_2_capacity = (
            max(0, int(section_2_frames.shape[0]) - 1)
            if section_2_frames is not None
            else 0
        )
        source_cuts = min(extra_cuts, section_1_capacity + section_2_capacity)
        section_1_cut, section_2_cut = _balanced_split(
            source_cuts,
            section_1_capacity,
            section_2_capacity,
        )

        # If a remove preference asks for more cuts than the two sections can
        # provide, reduce requested white frames by the remainder instead.
        blank_total = max(
            0,
            frames_to_add_between + extra_blank_frames
            - max(0, extra_cuts - source_cuts),
        )
        section_1_output = section_1_frames[: -section_1_cut or None]
        section_2_output = (
            section_2_frames[section_2_cut:]
            if section_2_frames is not None
            else None
        )

        output_sequences = []
        if start_frame is not None:
            output_sequences.append(start_frame)
        output_sequences.append(section_1_output)
        if section_2_output is not None:
            output_sequences.append(section_2_output)
        if end_frame is not None:
            output_sequences.append(end_frame)

        transition_weights = [
            _transition_distance(first[-1], second[0])
            for first, second in zip(output_sequences, output_sequences[1:])
        ]
        blank_allocations = _allocate_weighted(blank_total, transition_weights)

        frame_parts = []
        mask_parts = []
        for index, sequence in enumerate(output_sequences):
            frame_parts.append(sequence)
            mask_parts.append(
                torch.zeros(
                    (int(sequence.shape[0]), reference_shape[0], reference_shape[1]),
                    dtype=torch.float32,
                    device=section_1_frames.device,
                )
            )
            if index >= len(blank_allocations):
                continue
            blank_count = blank_allocations[index]
            frame_parts.append(
                torch.ones(
                    (blank_count, *reference_shape),
                    dtype=section_1_frames.dtype,
                    device=section_1_frames.device,
                )
            )
            mask_parts.append(
                torch.ones(
                    (blank_count, reference_shape[0], reference_shape[1]),
                    dtype=torch.float32,
                    device=section_1_frames.device,
                )
            )

        frames = torch.cat(frame_parts, dim=0)
        mask = torch.cat(mask_parts, dim=0)

        if int(frames.shape[0]) != final_total:
            raise RuntimeError(
                "Internal WAN frame calculation mismatch: "
                f"expected {final_total}, got {int(frames.shape[0])}."
            )

        return (frames, mask)


# Keep imports from the previous node implementation working for external code.
WANBeginEndFrames = WANConnectFrames
