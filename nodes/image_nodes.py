from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..utils import parse_hex_color


class ImageRotatorNode:
    """Rotate an image batch by the provided degrees with configurable background color."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rotated_image",)
    FUNCTION = "rotate_image"
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "degrees": (
                    "INT",
                    {
                        "default": 0,
                        "min": -360,
                        "max": 360,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "background_color": (
                    "STRING",
                    {"default": "#000000", "multiline": False},
                ),
            }
        }

    def rotate_image(self, images: torch.Tensor, degrees: int, background_color: str):
        bg_rgb = parse_hex_color(background_color)
        images_np = images.detach().cpu().numpy()

        rotated_images = []
        for img_data in images_np:
            img_255 = (img_data * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_255)
            rotated_pil = pil_img.rotate(-degrees, expand=False, fillcolor=bg_rgb)
            fitted = self._fit_to_size(rotated_pil, pil_img.size, bg_rgb)
            rotated_np = np.asarray(fitted, dtype=np.float32) / 255.0
            rotated_images.append(rotated_np)

        result = torch.from_numpy(np.stack(rotated_images))
        return (result,)

    @staticmethod
    def _fit_to_size(
        pil_img: Image.Image, target_size: tuple[int, int], bg_rgb: tuple[int, int, int]
    ):
        """Pad or crop image to the target size after rotation."""
        target_width, target_height = target_size
        fitted_img = Image.new("RGB", target_size, bg_rgb)
        left = (target_width - pil_img.width) // 2
        top = (target_height - pil_img.height) // 2
        fitted_img.paste(pil_img, (left, top))
        return fitted_img


class PoseImageSetupNode:
    """Expand an image around a mask and fill uncovered regions."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "setup_pose_images"
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
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
                "only_fill": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    def setup_pose_images(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        fill_color: str,
        width_change: int,
        height_change: int,
        offset_x: int,
        offset_y: int,
        only_fill: bool,
        mask: torch.Tensor = None,
    ):
        fill_rgb = parse_hex_color(fill_color)
        images_np = images.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        if mask_np.ndim == 3:
            mask_combined = (mask_np > 0.5).any(axis=0)
        # If no mask is provided, use the entire image
        if mask is None:
            img_height, img_width = images_np.shape[1:3]
            mask_combined = np.ones((img_height, img_width), dtype=bool)
        else:
            mask_combined = mask_np > 0.5
            mask_np = mask.detach().cpu().numpy()
            if mask_np.ndim == 3:
                mask_combined = (mask_np > 0.5).any(axis=0)
            else:
                mask_combined = mask_np > 0.5

        if not np.any(mask_combined):
            # Nothing selected—return images unchanged
            return (images,)
            if not np.any(mask_combined):
                # Nothing selected—return images unchanged
                return (images,)

        mask_y_indices, mask_x_indices = np.nonzero(mask_combined)
        min_y, max_y = mask_y_indices.min(), mask_y_indices.max() + 1
        min_x, max_x = mask_x_indices.min(), mask_x_indices.max() + 1

        base_width = max_x - min_x
        base_height = max_y - min_y

        # Determine symmetric expansion based on combined width/height adjustments
        left_expand = width_change // 2
        top_expand = height_change // 2

        expanded_width = max(1, base_width + width_change)
        expanded_height = max(1, base_height + height_change)

        # Determine source crop bounds in the original frame
        start_x = int(min_x - left_expand - offset_x)
        start_y = int(min_y - top_expand - offset_y)
        end_x = start_x + expanded_width
        end_y = start_y + expanded_height

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

            canvas = np.empty(
                (expanded_height, expanded_width, channels), dtype=np.uint8
            )
            canvas[...] = fill_values

            if only_fill:
                # Copy only the masked pixels; everything else stays filled.
                valid_mask = (
                    (mask_x_indices >= start_x)
                    & (mask_x_indices < end_x)
                    & (mask_y_indices >= start_y)
                    & (mask_y_indices < end_y)
                )
                if np.any(valid_mask):
                    src_x = mask_x_indices[valid_mask]
                    src_y = mask_y_indices[valid_mask]
                    dst_x = src_x - start_x
                    dst_y = src_y - start_y
                    canvas[dst_y, dst_x] = img_255[src_y, src_x]
            else:
                # Clamp crop bounds to original frame and determine placement on canvas
                src_x0 = max(0, start_x)
                src_y0 = max(0, start_y)
                src_x1 = min(img_255.shape[1], end_x)
                src_y1 = min(img_255.shape[0], end_y)

                if src_x0 < src_x1 and src_y0 < src_y1:
                    dst_x0 = max(0, -start_x)
                    dst_y0 = max(0, -start_y)
                    dst_x1 = dst_x0 + (src_x1 - src_x0)
                    dst_y1 = dst_y0 + (src_y1 - src_y0)

                    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = img_255[
                        src_y0:src_y1, src_x0:src_x1
                    ]

            result_images.append(canvas.astype(np.float32) / 255.0)

        result = torch.from_numpy(np.stack(result_images))

        return (result,)
