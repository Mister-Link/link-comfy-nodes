from __future__ import annotations

import os
import tempfile
import uuid

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None  # type: ignore


class FastImagePreviewNode:
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_images"
    CATEGORY = "image/preview"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    def preview_images(self, images: torch.Tensor):
        if not isinstance(images, torch.Tensor):
            return {"ui": {"fast_images": []}}

        batch_size = images.shape[0]
        if batch_size == 0:
            return {"ui": {"fast_images": []}}

        if folder_paths:
            temp_dir = folder_paths.get_temp_directory()
        else:
            temp_dir = tempfile.gettempdir()

        THUMB_SIZE = 256
        THUMB_QUALITY = 70
        FULL_SIZE = 2048
        FULL_QUALITY = 95

        preview_data = []
        unique_id = uuid.uuid4().hex[:8]
        for idx, img_tensor in enumerate(images):
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            full_filename = f"fast_preview_{unique_id}_{idx:04d}_full.webp"
            full_filepath = os.path.join(temp_dir, full_filename)
            pil_full = pil_img.copy()
            pil_full.thumbnail((FULL_SIZE, FULL_SIZE), Image.Resampling.LANCZOS)
            pil_full.save(full_filepath, format="WEBP", quality=FULL_QUALITY, method=6)

            pil_thumb = pil_img.copy()
            pil_thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)

            thumb_filename = f"fast_preview_{unique_id}_{idx:04d}.webp"
            thumb_filepath = os.path.join(temp_dir, thumb_filename)
            pil_thumb.save(
                thumb_filepath, format="WEBP", quality=THUMB_QUALITY, method=4
            )

            preview_data.append(
                {
                    "filename": thumb_filename,
                    "full_filename": full_filename,
                    "subfolder": "",
                    "type": "temp",
                    "width": pil_full.width,
                    "height": pil_full.height,
                }
            )

        return {"ui": {"fast_images": preview_data}}
