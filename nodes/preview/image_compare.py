from __future__ import annotations

import base64
import gc
import io

import numpy as np
from PIL import Image


class ImageCompareNode:
    """Side-by-side slider comparison of two images (ported from Sean-Bradley/ComfyUI-Image-Compare)."""

    CATEGORY = "image/preview"
    RETURN_TYPES = ()
    FUNCTION = "compare"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            }
        }

    def compare(self, image_a, image_b):
        if image_a is None or len(image_a) == 0:
            return {}
        if image_b is None or len(image_b) == 0:
            return {}

        try:
            def tensor_to_pil(img_tensor):
                img = (img_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                return Image.fromarray(img)

            pil_img_a = tensor_to_pil(image_a)
            pil_img_b = tensor_to_pil(image_b)

            b64_a = self._pil_to_base64_chunks(pil_img_a)
            b64_b = self._pil_to_base64_chunks(pil_img_b)

            pil_img_a.close()
            pil_img_b.close()
            del pil_img_a, pil_img_b

            return {
                "ui": {
                    "b64_a": b64_a,
                    "b64_b": b64_b,
                }
            }
        finally:
            gc.collect()

    @staticmethod
    def _pil_to_base64_chunks(img: Image.Image, chunk_size: int = 65536):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()
        return [encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)]
