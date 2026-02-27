from __future__ import annotations

import json
import numpy as np
import torch


class CropByBBoxNode:
    """Crops a set of frames using bounding box coordinates from AutoCropper."""

    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("cropped_frames", "cropped_alpha", "width", "height")
    FUNCTION = "crop_by_bbox"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "bbox": ("STRING", {"default": "{}"}),
                "padding": (
                    "INT",
                    {"default": 0, "min": 0, "max": 500, "step": 1},
                ),
                "padding_color": (
                    "STRING",
                    {"default": "#000000"},
                ),
                "use_image_padding": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If true, padding area is taken from the original frame when in bounds; "
                            "out-of-bounds area uses padding_color."
                        ),
                    },
                ),
            },
            "optional": {
                "alpha": ("MASK",),
            },
        }

    def _hex_to_rgb(self, hex_color: str):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r, g, b = (
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16),
            )
            return (r / 255.0, g / 255.0, b / 255.0)
        return (0.0, 0.0, 0.0)

    def crop_by_bbox(
        self,
        frames: torch.Tensor,
        bbox: str,
        padding: int,
        padding_color: str,
        use_image_padding: bool,
        alpha: torch.Tensor = None,
    ):
        # Parse bbox JSON
        try:
            bbox_data = json.loads(bbox)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid bbox JSON: {bbox}")

        bbox_x = bbox_data.get("x", 0)
        bbox_y = bbox_data.get("y", 0)
        bbox_w = bbox_data.get("w", 256)
        bbox_h = bbox_data.get("h", 256)

        frames_np = frames.cpu().numpy()

        if frames_np.ndim != 4:
            raise ValueError(
                f"Expected frames with shape (N, H, W, C), got {frames_np.shape}"
            )

        if alpha is not None:
            alpha_np = alpha.cpu().numpy()
            if alpha_np.ndim == 4 and alpha_np.shape[-1] == 1:
                alpha_np = alpha_np[..., 0]
            if alpha_np.ndim != 3:
                raise ValueError(
                    f"Expected alpha with shape (N, H, W), got {alpha_np.shape}"
                )
        else:
            alpha_np = np.ones(
                (frames_np.shape[0], frames_np.shape[1], frames_np.shape[2]),
                dtype=np.float32,
            )

        num_frames, H, W, C = frames_np.shape

        # Clamp bbox to frame boundaries
        x1 = max(0, bbox_x)
        y1 = max(0, bbox_y)
        x2 = min(W, bbox_x + bbox_w)
        y2 = min(H, bbox_y + bbox_h)

        print(
            f"[CropByBBox] Cropping {num_frames} frames to ({x1}, {y1}) -> ({x2}, {y2}), "
            f"size: {x2 - x1}x{y2 - y1}"
        )

        cropped_frames = []
        cropped_alphas = []

        for i in range(num_frames):
            frame = frames_np[i]
            alpha_frame = alpha_np[i]

            cropped_frame = frame[y1:y2, x1:x2]
            cropped_alpha = alpha_frame[y1:y2, x1:x2]

            if padding > 0:
                crop_width = x2 - x1
                crop_height = y2 - y1
                padded_h = crop_height + (padding * 2)
                padded_w = crop_width + (padding * 2)

                r, g, b = self._hex_to_rgb(padding_color)

                padded_frame = np.zeros(
                    (padded_h, padded_w, C), dtype=cropped_frame.dtype
                )
                if C >= 3:
                    padded_frame[:, :, 0] = r
                    padded_frame[:, :, 1] = g
                    padded_frame[:, :, 2] = b
                if C == 4:
                    padded_frame[:, :, 3] = 0.0

                padded_alpha = np.zeros((padded_h, padded_w), dtype=cropped_alpha.dtype)

                if use_image_padding:
                    src_x1 = x1 - padding
                    src_y1 = y1 - padding
                    src_x2 = x2 + padding
                    src_y2 = y2 + padding

                    # Copy only the in-bounds part from the original frame.
                    ix1 = max(0, src_x1)
                    iy1 = max(0, src_y1)
                    ix2 = min(W, src_x2)
                    iy2 = min(H, src_y2)

                    if ix2 > ix1 and iy2 > iy1:
                        dx1 = ix1 - src_x1
                        dy1 = iy1 - src_y1
                        dx2 = dx1 + (ix2 - ix1)
                        dy2 = dy1 + (iy2 - iy1)
                        padded_frame[dy1:dy2, dx1:dx2] = frame[iy1:iy2, ix1:ix2]
                        padded_alpha[dy1:dy2, dx1:dx2] = alpha_frame[iy1:iy2, ix1:ix2]
                else:
                    crop_width = x2 - x1
                    crop_height = y2 - y1
                    padded_frame[
                        padding : padding + crop_height, padding : padding + crop_width
                    ] = cropped_frame
                    padded_alpha[
                        padding : padding + crop_height, padding : padding + crop_width
                    ] = cropped_alpha

                cropped_frames.append(padded_frame)
                cropped_alphas.append(padded_alpha)
            else:
                cropped_frames.append(cropped_frame)
                cropped_alphas.append(cropped_alpha)

        result_frames = torch.from_numpy(np.stack(cropped_frames).astype(np.float32))
        result_alphas = torch.from_numpy(np.stack(cropped_alphas).astype(np.float32))

        final_width = result_frames.shape[2]
        final_height = result_frames.shape[1]

        print(
            f"[CropByBBox] Done. Output size: {final_width}x{final_height} "
            f"(with {padding}px padding)"
        )

        return (result_frames, result_alphas, final_width, final_height)
