from __future__ import annotations

import json

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
    """Expand an image around a bbox region and fill uncovered areas."""

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
        images: torch.Tensor,
        fill_with_color: bool,
        fill_color: str,
        width_change: int,
        height_change: int,
        offset_x: int,
        offset_y: int,
    ):
        fill_rgb = parse_hex_color(fill_color)
        images_np = images.detach().cpu().numpy()
        img_height, img_width = images_np.shape[1:3]

        # Use full image dimensions
        bbox_x = 0
        bbox_y = 0
        bbox_width = img_width
        bbox_height = img_height

        # Calculate final canvas size by expanding the bbox dimensions
        final_width = max(1, bbox_width + width_change)
        final_height = max(1, bbox_height + height_change)

        # Calculate padding amounts (split evenly on each side)
        left_pad = width_change // 2
        right_pad = width_change - left_pad
        top_pad = height_change // 2
        bottom_pad = height_change - top_pad

        # Process images
        result_images = []

        for img_idx, img_data in enumerate(images_np):
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

            # Create canvas filled with background color
            canvas = np.full(
                (final_height, final_width, channels), fill_values, dtype=np.uint8
            )

            # Calculate where the bbox region should be placed on the canvas
            # The bbox expands equally in all directions, so we add padding
            canvas_bbox_x = left_pad + offset_x
            canvas_bbox_y = top_pad + offset_y

            # Extract the bbox region from the original image
            # Clamp to image boundaries
            src_x0 = max(0, bbox_x)
            src_y0 = max(0, bbox_y)
            src_x1 = min(img_width, bbox_x + bbox_width)
            src_y1 = min(img_height, bbox_y + bbox_height)

            # Calculate destination position on canvas
            dst_x0 = canvas_bbox_x
            dst_y0 = canvas_bbox_y
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)

            # Clamp destination to canvas boundaries
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

            # Place the bbox region on the canvas
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


class CropToContentNode:
    """Crop images to non-transparent content with a 1-pixel border."""

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "alpha")
    FUNCTION = "crop_to_content"
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "alpha": ("MASK",),
            },
        }

    def crop_to_content(self, images: torch.Tensor, alpha: torch.Tensor | None = None):
        frames = images.detach().cpu().float()
        if frames.ndim != 4:
            raise ValueError("Expected images with shape (N, H, W, C)")

        has_alpha = frames.shape[-1] == 4
        if alpha is not None:
            mask = alpha.detach().cpu().float()
            if mask.ndim == 4 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.ndim != 3:
                raise ValueError("Expected alpha mask with shape (N, H, W)")
            if mask.shape[0] != frames.shape[0]:
                raise ValueError("Alpha mask batch size does not match images")
            alpha_mask = mask.clamp(0, 1)
        elif has_alpha:
            alpha_mask = frames[..., 3].clamp(0, 1)
        else:
            alpha_mask = torch.ones(
                frames.shape[0],
                frames.shape[1],
                frames.shape[2],
                device=frames.device,
                dtype=frames.dtype,
            )

        active = alpha_mask > 0.0
        if active.any():
            coords = active.nonzero(as_tuple=False)
            height = alpha_mask.shape[1]
            width = alpha_mask.shape[2]
            y_min = int(coords[:, 1].min().item())
            x_min = int(coords[:, 2].min().item())
            y_max = int(coords[:, 1].max().item())
            x_max = int(coords[:, 2].max().item())
            y_min = max(y_min - 1, 0)
            x_min = max(x_min - 1, 0)
            y_max = min(y_max + 2, height)
            x_max = min(x_max + 2, width)
            frames = frames[:, y_min:y_max, x_min:x_max, :]
            alpha_mask = alpha_mask[:, y_min:y_max, x_min:x_max]

        return (frames, alpha_mask)


class PixelationDimensionsNode:
    """Provide width/height presets for pixelation targets."""

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_dimensions"
    CATEGORY = "image/transform"

    _PRESETS = {
        "Spirie": (979, 1562),
        "Custom": (0, 0),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls._PRESETS.keys()),),
                "custom_width": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 8192, "step": 1},
                ),
                "custom_height": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 8192, "step": 1},
                ),
            }
        }

    def get_dimensions(self, preset: str, custom_width: int, custom_height: int):
        if preset == "Custom":
            width, height = custom_width, custom_height
        else:
            width, height = self._PRESETS[preset]
        return (width, height)


class SpritesheetBuilderNode:
    """Combine frames into a spritesheet with a target aspect ratio."""

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("spritesheet", "alpha", "metadata")
    FUNCTION = "build_spritesheet"
    CATEGORY = "image/transform"

    _ASPECT_RATIOS = {
        "1:1 (Square)": (1, 1),
        "4:3 (Landscape)": (4, 3),
        "3:4 (Portrait)": (3, 4),
        "16:9 (Landscape)": (16, 9),
        "9:16 (Portrait)": (9, 16),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "alpha": ("MASK",),
                "aspect_ratio": (list(cls._ASPECT_RATIOS.keys()),),
            }
        }

    def build_spritesheet(
        self, frames: torch.Tensor, alpha: torch.Tensor, aspect_ratio: str
    ):
        frames_cpu = frames.detach().cpu().float()
        if frames_cpu.ndim != 4:
            raise ValueError("Expected frames with shape (N, H, W, C)")

        if alpha is not None:
            alpha_cpu = alpha.detach().cpu().float()
            if alpha_cpu.ndim == 4 and alpha_cpu.shape[-1] == 1:
                alpha_cpu = alpha_cpu[..., 0]
            if alpha_cpu.ndim != 3:
                raise ValueError("Expected alpha mask with shape (N, H, W)")
            if alpha_cpu.shape[0] != frames_cpu.shape[0]:
                raise ValueError("Alpha mask batch size does not match frames")
            alpha_cpu = alpha_cpu.clamp(0, 1)
        else:
            alpha_cpu = None

        target_ratio = self._aspect_ratio_value(aspect_ratio)
        frame_count, frame_height, frame_width, frame_channels = frames_cpu.shape
        use_alpha = alpha_cpu is not None or frame_channels == 4

        if use_alpha and frame_channels == 3:
            alpha_stack = (
                alpha_cpu
                if alpha_cpu is not None
                else torch.ones(
                    (frame_count, frame_height, frame_width),
                    dtype=frames_cpu.dtype,
                )
            )
            frames_cpu = torch.cat((frames_cpu, alpha_stack.unsqueeze(-1)), dim=-1)
            frame_channels = 4
        elif not use_alpha and frame_channels == 4:
            frames_cpu = frames_cpu[..., :3]
            frame_channels = 3

        columns, rows = self._closest_grid(
            frame_count, target_ratio, frame_width, frame_height
        )
        sheet_width = frame_width * columns
        sheet_height = frame_height * rows

        spritesheet = np.zeros(
            (sheet_height, sheet_width, frame_channels), dtype=np.float32
        )

        frames_np = frames_cpu.numpy()
        for idx in range(frame_count):
            row = idx // columns
            col = idx % columns
            y0 = row * frame_height
            x0 = col * frame_width
            spritesheet[
                y0 : y0 + frame_height, x0 : x0 + frame_width, :frame_channels
            ] = frames_np[idx, :, :, :frame_channels]

        result = torch.from_numpy(spritesheet).unsqueeze(0)
        if frame_channels == 4:
            alpha_mask = result[..., 3].clone()
        else:
            alpha_mask = torch.ones((1, sheet_height, sheet_width), dtype=result.dtype)
        metadata = {
            "spritesheet": {
                "width": sheet_width,
                "height": sheet_height,
                "rows": rows,
                "columns": columns,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "frame_count": frame_count,
            }
        }
        return (result, alpha_mask, json.dumps(metadata, indent=2))

    @classmethod
    def _aspect_ratio_value(cls, aspect_ratio: str) -> float:
        width, height = cls._ASPECT_RATIOS[aspect_ratio]
        return width / height

    @staticmethod
    def _closest_grid(
        frame_count: int, target_ratio: float, frame_width: int, frame_height: int
    ) -> tuple[int, int]:
        best_cols = 1
        best_rows = frame_count
        best_diff = float("inf")
        best_cells = frame_count

        for rows in range(1, frame_count + 1):
            if frame_count % rows != 0:
                continue
            cols = frame_count // rows
            ratio = (cols * frame_width) / (rows * frame_height)
            diff = abs(ratio - target_ratio)
            cells = rows * cols
            if (
                diff < best_diff
                or (diff == best_diff and cells < best_cells)
                or (diff == best_diff and cells == best_cells and cols > best_cols)
            ):
                best_diff = diff
                best_cols = cols
                best_rows = rows
                best_cells = cells

        return best_cols, best_rows
