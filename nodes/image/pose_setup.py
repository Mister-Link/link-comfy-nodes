from __future__ import annotations

import numpy as np
import torch

from ...utils import parse_hex_color


class PoseImageSetupNode:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "setup_pose_images"
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fill_with_color": ("BOOLEAN", {"default": True}),
                "fill_color": ("STRING", {"default": "#000000", "multiline": False}),
                "width_change": (
                    "INT",
                    {"default": 0, "min": -2048, "max": 2048, "step": 1},
                ),
                "height_change": (
                    "INT",
                    {"default": 0, "min": -2048, "max": 2048, "step": 1},
                ),
                "offset_x": (
                    "INT",
                    {"default": 0, "min": -2048, "max": 2048, "step": 1},
                ),
                "offset_y": (
                    "INT",
                    {"default": 0, "min": -2048, "max": 2048, "step": 1},
                ),
            },
        }

    def setup_pose_images(
        self,
        images,
        fill_with_color: bool,
        fill_color: str,
        width_change: int,
        height_change: int,
        offset_x: int,
        offset_y: int,
    ):
        _ = fill_with_color
        fill_rgb = parse_hex_color(fill_color)
        images_np = images.detach().cpu().numpy()
        img_height, img_width = images_np.shape[1:3]

        bbox_x = 0
        bbox_y = 0
        bbox_width = img_width
        bbox_height = img_height

        final_width = max(1, bbox_width + width_change)
        final_height = max(1, bbox_height + height_change)

        left_pad = width_change // 2
        top_pad = height_change // 2

        result_images = []

        for img_data in images_np:
            img_255 = (img_data * 255).astype(np.uint8)
            if img_255.ndim == 2:
                img_255 = img_255[:, :, None]

            channels = img_255.shape[2]
            fill_values = np.array(fill_rgb, dtype=np.uint8)
            if channels == 1:
                fill_values = fill_values[:1]
            elif channels > fill_values.shape[0]:
                fill_values = np.pad(
                    fill_values, (0, channels - fill_values.shape[0]), mode="edge"
                )
            else:
                fill_values = fill_values[:channels]

            canvas = np.full(
                (final_height, final_width, channels), fill_values, dtype=np.uint8
            )

            canvas_bbox_x = left_pad + offset_x
            canvas_bbox_y = top_pad + offset_y

            src_x0 = max(0, bbox_x)
            src_y0 = max(0, bbox_y)
            src_x1 = min(img_width, bbox_x + bbox_width)
            src_y1 = min(img_height, bbox_y + bbox_height)

            dst_x0 = canvas_bbox_x
            dst_y0 = canvas_bbox_y
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)

            if dst_x0 < 0:
                src_x0 -= dst_x0
                dst_x0 = 0
            if dst_y0 < 0:
                src_y0 -= dst_y0
                dst_y0 = 0
            if dst_x1 > final_width:
                src_x1 -= dst_x1 - final_width
                dst_x1 = final_width
            if dst_y1 > final_height:
                src_y1 -= dst_y1 - final_height
                dst_y1 = final_height

            if (
                src_x1 > src_x0
                and src_y1 > src_y0
                and dst_x1 > dst_x0
                and dst_y1 > dst_y0
            ):
                canvas[dst_y0:dst_y1, dst_x0:dst_x1] = img_255[
                    src_y0:src_y1, src_x0:src_x1
                ]

            result_images.append(canvas.astype(np.float32) / 255.0)

        result_tensor = torch.from_numpy(np.stack(result_images))

        return (result_tensor,)
