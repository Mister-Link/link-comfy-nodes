from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ...utils import parse_hex_color


class ImageRotatorNode:
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

    def rotate_image(self, images, degrees: int, background_color: str):
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
    def _fit_to_size(pil_img, target_size, bg_rgb):
        target_width, target_height = target_size
        fitted_img = Image.new("RGB", target_size, bg_rgb)
        left = (target_width - pil_img.width) // 2
        top = (target_height - pil_img.height) // 2
        fitted_img.paste(pil_img, (left, top))
        return fitted_img
