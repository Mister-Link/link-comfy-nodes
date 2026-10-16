"""Build and restore an explicit WAN loop-seam replacement window."""

from __future__ import annotations

import json

import torch


BRIDGE_METADATA_FORMAT = "link-comfy-nodes/wan-loop-bridge-v2"


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


class WANLoopBridge:
    """Replace a circular seam with an explicit generated frame window."""

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
        "Context before the seam, masked replacement frames, and context after the seam.",
        "White only for frames that will be inserted into the final animation; sampler padding is black.",
        "Final full-animation frame count after skipping seam frames and inserting the requested frames.",
        "Metadata for WAN Unloop Bridge to remove the seam window and insert only generated frames.",
        "Actual WAN-valid frame count of the sampling segment.",
        "Untouched original animation batch for WAN Unloop Bridge.",
    )
    FUNCTION = "create"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "IMAGE",
                    {
                        "tooltip": "Full source animation containing the seam to replace.",
                    },
                ),
                "skip_end_frames": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 9998,
                        "step": 1,
                        "tooltip": (
                            "Raw frames skipped at the end of the animation before the generated "
                            "bridge; context is taken immediately before these frames."
                        ),
                    },
                ),
                "skip_begin_frames": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 9998,
                        "step": 1,
                        "tooltip": (
                            "Raw frames skipped at the beginning of the animation after the generated "
                            "bridge; context is taken immediately after these frames."
                        ),
                    },
                ),
                "insert_frames": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 9998,
                        "step": 1,
                        "tooltip": (
                            "Number of generated frames inserted into the skipped seam; "
                            "sampler-only padding is added separately when needed for a WAN-valid segment."
                        ),
                    },
                ),
                "context_frames": (
                    "INT",
                    {
                        "default": 4,
                        "min": 2,
                        "max": 9998,
                        "step": 1,
                        "tooltip": (
                            "Total source guide frames outside the replacement window; "
                            "odd values give the extra frame to the before side."
                        ),
                    },
                ),
            }
        }

    def create(
        self,
        frames: torch.Tensor,
        skip_end_frames: int = 4,
        skip_begin_frames: int = 4,
        insert_frames: int = 4,
        context_frames: int = 4,
    ):
        frames = _as_image_batch(frames)
        frame_count = int(frames.shape[0])
        skip_end_frames = int(skip_end_frames)
        skip_begin_frames = int(skip_begin_frames)
        insert_frames = int(insert_frames)
        context_frames = int(context_frames)

        if skip_end_frames < 0:
            raise ValueError("skip_end_frames must be non-negative.")
        if skip_begin_frames < 0:
            raise ValueError("skip_begin_frames must be non-negative.")
        if insert_frames < 1:
            raise ValueError("insert_frames must be at least one.")
        if context_frames < 2:
            raise ValueError(
                "context_frames must be at least 2 so the bridge has source context "
                "on both sides of the seam."
            )
        skipped_frame_count = skip_end_frames + skip_begin_frames
        if skipped_frame_count >= frame_count:
            raise ValueError(
                "skip_end_frames + skip_begin_frames must leave at least one source frame; "
                f"received {skipped_frame_count} for {frame_count} input frames."
            )

        source_frame_count = frame_count - skipped_frame_count
        if context_frames > source_frame_count:
            raise ValueError(
                f"context_frames={context_frames} is too large after skipping "
                f"{skipped_frame_count} seam frames; only {source_frame_count} "
                "source frames remain."
            )

        # Keep the requested insertion count exact. WAN validity applies to the
        # sampled bridge segment, not necessarily to the restored final batch.
        target_total = source_frame_count + insert_frames

        # Skip explicit end and beginning ranges to define the seam window.
        keep_begin = skip_begin_frames
        keep_end = frame_count - skip_end_frames

        context_before_frames = (context_frames + 1) // 2
        context_after_frames = context_frames // 2
        before_context = frames[
            keep_end - context_before_frames : keep_end
        ]
        after_context = frames[keep_begin : keep_begin + context_after_frames]

        height, width, channels = (
            int(frames.shape[1]),
            int(frames.shape[2]),
            int(frames.shape[3]),
        )
        blank_frames = torch.ones(
            (insert_frames, height, width, channels),
            dtype=frames.dtype,
            device=frames.device,
        )

        base_segment_count = (
            context_before_frames
            + insert_frames
            + context_after_frames
        )
        segment_padding_count = (1 - base_segment_count) % 4
        padding_before_count = (segment_padding_count + 1) // 2
        padding_after_count = segment_padding_count // 2

        def _padding_slice(start: int, end: int, count: int, fallback: torch.Tensor):
            if count < 1:
                return frames[:0]
            part = frames[max(start, keep_begin) : min(end, keep_end)]
            if int(part.shape[0]) >= count:
                return part[:count]
            if int(part.shape[0]) == 0:
                return fallback[:1].expand(count, -1, -1, -1)
            return torch.cat(
                (
                    part,
                    fallback[:1].expand(count - int(part.shape[0]), -1, -1, -1),
                ),
                dim=0,
            )

        # Keep sampler-only padding unmasked and outside the requested context.
        # If padding is odd, the beginning receives the extra frame.
        padding_before = _padding_slice(
            keep_end - context_before_frames - padding_before_count,
            keep_end - context_before_frames,
            padding_before_count,
            before_context,
        )
        padding_after = _padding_slice(
            keep_begin + context_after_frames,
            keep_begin + context_after_frames + padding_after_count,
            padding_after_count,
            after_context,
        )

        segment_frames = torch.cat(
            (padding_before, before_context, blank_frames, after_context, padding_after),
            dim=0,
        )
        segment_mask = torch.cat(
            (
                torch.zeros(
                    (int(padding_before.shape[0]), height, width),
                    dtype=torch.float32,
                    device=frames.device,
                ),
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
                torch.zeros(
                    (int(padding_after.shape[0]), height, width),
                    dtype=torch.float32,
                    device=frames.device,
                ),
            ),
            dim=0,
        )

        segment_frame_count = int(segment_frames.shape[0])
        if (segment_frame_count - 1) % 4 != 0:
            raise RuntimeError(
                "WAN Loop Bridge could not resolve the sampling segment to a valid 1 + 4n length."
            )
        metadata = {
            "format": BRIDGE_METADATA_FORMAT,
            "input_frame_count": frame_count,
            "output_frame_count": target_total,
            "segment_frame_count": segment_frame_count,
            "skip_end_frames": skip_end_frames,
            "skip_begin_frames": skip_begin_frames,
            "insert_frames_requested": insert_frames,
            "insert_frames": insert_frames,
            "context_frames": context_frames,
            "context_before_frames": context_before_frames,
            "context_after_frames": context_after_frames,
            "padding_before_frames": int(padding_before.shape[0]),
            "padding_after_frames": int(padding_after.shape[0]),
            "keep_begin": keep_begin,
            "keep_end": keep_end,
            "source_frame_count": source_frame_count,
            "insert_index": source_frame_count,
            "generated_segment_offset": int(padding_before.shape[0]) + context_before_frames,
            "segment_padding_count": segment_padding_count,
            "output_frame_count_wan_valid": (target_total - 1) % 4 == 0,
        }

        return (
            segment_frames,
            segment_mask,
            target_total,
            json.dumps(metadata, separators=(",", ":")),
            segment_frame_count,
            frames,
        )


class WANUnloopBridge:
    """Restore a WAN Loop Bridge segment into the surviving raw sequence."""

    CATEGORY = "conditioning/video_models"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    OUTPUT_TOOLTIPS = (
        "Surviving raw frames with only the generated seam frames inserted.",
    )
    FUNCTION = "restore"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_frames": (
                    "IMAGE",
                    {"tooltip": "Untouched source animation passed through WAN Loop Bridge."},
                ),
                "inpainted_frames": (
                    "IMAGE",
                    {"tooltip": "Complete WAN-valid bridge segment after sampling."},
                ),
                "metadata": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Metadata output from WAN Loop Bridge.",
                    },
                ),
            }
        }

    @staticmethod
    def _parse_metadata(metadata_text: str) -> dict:
        try:
            metadata = json.loads(metadata_text)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("metadata is not valid WAN Loop Bridge JSON.") from error
        if not isinstance(metadata, dict) or metadata.get("format") != BRIDGE_METADATA_FORMAT:
            raise ValueError("metadata is not recognized WAN Loop Bridge metadata.")
        required = (
            "input_frame_count",
            "output_frame_count",
            "segment_frame_count",
            "keep_begin",
            "keep_end",
            "insert_index",
            "generated_segment_offset",
            "insert_frames",
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
        if raw_count != int(metadata["input_frame_count"]):
            raise ValueError(
                "WAN Unloop Bridge requires the untouched raw batch: "
                f"metadata expects {metadata['input_frame_count']}, received {raw_count}."
            )
        if segment_count != int(metadata["segment_frame_count"]):
            raise ValueError(
                "WAN Unloop Bridge received the wrong sampled segment length: "
                f"metadata expects {metadata['segment_frame_count']}, received {segment_count}."
            )
        if tuple(inpainted_frames.shape[1:]) != tuple(raw_frames.shape[1:]):
            raise ValueError(
                "raw_frames and inpainted_frames must have matching height, width, and channels."
            )

        keep_begin = int(metadata["keep_begin"])
        keep_end = int(metadata["keep_end"])
        if keep_begin < 0 or keep_end > raw_count or keep_begin >= keep_end:
            raise ValueError("metadata contains invalid raw keep boundaries.")

        source_frames = raw_frames[keep_begin:keep_end]
        source_count = int(source_frames.shape[0])
        insert_index = int(metadata["insert_index"])
        if insert_index < 0 or insert_index > source_count:
            raise ValueError("metadata contains an invalid insertion index.")

        generated_offset = int(metadata["generated_segment_offset"])
        generated_count = int(metadata["insert_frames"])
        if generated_offset < 0 or generated_count < 1:
            raise ValueError("metadata contains an invalid generated frame slice.")
        if generated_offset + generated_count > segment_count:
            raise ValueError("metadata points outside the sampled bridge segment.")

        generated_frames = inpainted_frames[
            generated_offset : generated_offset + generated_count
        ].to(device=source_frames.device, dtype=source_frames.dtype)
        restored_frames = torch.cat(
            (
                source_frames[:insert_index],
                generated_frames,
                source_frames[insert_index:],
            ),
            dim=0,
        )
        expected_output_count = int(metadata["output_frame_count"])
        if int(restored_frames.shape[0]) != expected_output_count:
            raise RuntimeError(
                "WAN Unloop Bridge produced an unexpected frame count: "
                f"expected {expected_output_count}, got {restored_frames.shape[0]}."
            )

        return (restored_frames,)
