from __future__ import annotations

import torch


class AverageMaskRegionSizeNode:
    """Compute average bounding-box dimensions for mask content across frames."""

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("avg_width", "avg_height", "frames_used")
    FUNCTION = "calculate"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Calculates the average width and height of the white/non-zero mask region "
        "across a batch of mask frames."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "ignore_empty_frames": ("BOOLEAN", {"default": True}),
            },
        }

    def _normalize_masks(self, masks: torch.Tensor) -> torch.Tensor:
        if masks.ndim == 2:
            return masks.unsqueeze(0)

        if masks.ndim == 4 and masks.shape[-1] == 1:
            return masks[..., 0]

        if masks.ndim != 3:
            raise ValueError(
                f"Expected masks with shape (T, H, W), got {tuple(masks.shape)}"
            )

        return masks

    def calculate(
        self,
        masks: torch.Tensor,
        threshold: float = 0.5,
        ignore_empty_frames: bool = True,
    ):
        masks = self._normalize_masks(masks)

        widths: list[int] = []
        heights: list[int] = []

        for frame_mask in masks:
            y_indices, x_indices = torch.nonzero(frame_mask > threshold, as_tuple=True)

            if y_indices.numel() == 0 or x_indices.numel() == 0:
                if ignore_empty_frames:
                    continue
                widths.append(0)
                heights.append(0)
                continue

            min_y = int(y_indices.min().item())
            max_y = int(y_indices.max().item())
            min_x = int(x_indices.min().item())
            max_x = int(x_indices.max().item())

            widths.append(max_x - min_x + 1)
            heights.append(max_y - min_y + 1)

        frames_used = len(widths)
        if frames_used == 0:
            print("[AverageMaskRegionSizeNode] no mask content found")
            return (0, 0, 0)

        avg_width = int(round(sum(widths) / frames_used))
        avg_height = int(round(sum(heights) / frames_used))

        print(
            "[AverageMaskRegionSizeNode] "
            f"frames_used={frames_used}, avg_width={avg_width}, avg_height={avg_height}"
        )

        return (avg_width, avg_height, frames_used)
