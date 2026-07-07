from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans


class ImagePixelateNode:
    """Pixelate images via downscaling + k-means color quantization, with optional dithering and alpha-aware masking."""

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask")
    FUNCTION = "image_pixelate"
    CATEGORY = "image/transform"

    _ALPHA_FIT_THRESHOLD = 0.5
    _MAX_ITERATIONS = 100

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "pixelation_size": ("INT", {"default": 164, "min": 16, "max": 480, "step": 1}),
                "num_colors": ("INT", {"default": 16, "min": 2, "max": 256, "step": 1}),
                "init_mode": (["k-means++", "random", "none"],),
                "dither": (["False", "True"],),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    def image_pixelate(
        self,
        images: torch.Tensor,
        pixelation_size: int = 164,
        num_colors: int = 16,
        init_mode: str = "random",
        mask: torch.Tensor | None = None,
        dither: str = "False",
    ):
        dither_on = dither == "True"

        images_np = images.detach().cpu().numpy()
        has_alpha_channel = images_np.shape[-1] == 4
        pil_images = [Image.fromarray((img[..., :3] * 255).astype(np.uint8)) for img in images_np]
        alpha_images = self._resolve_alpha_images(images_np, mask, has_alpha_channel)

        pixelated, pixelated_alpha = self._pixelate_batch(
            pil_images,
            alpha_images,
            pixelation_size,
            num_colors,
            init_mode,
            dither_on,
        )

        result = torch.from_numpy(np.stack([np.asarray(img, dtype=np.float32) / 255.0 for img in pixelated]))
        result_mask = torch.from_numpy(
            np.stack([np.asarray(alpha, dtype=np.float32) / 255.0 for alpha in pixelated_alpha])
        )
        return (result, result_mask)

    @staticmethod
    def _resolve_alpha_images(
        images_np: np.ndarray, mask: torch.Tensor | None, has_alpha_channel: bool
    ) -> list[Image.Image]:
        batch, height, width = images_np.shape[0], images_np.shape[1], images_np.shape[2]

        if mask is not None:
            mask_np = mask.detach().cpu().numpy()
            if mask_np.ndim == 4 and mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]
            if mask_np.shape[0] != batch:
                raise ValueError(f"Mask batch size ({mask_np.shape[0]}) does not match images ({batch})")
            if mask_np.shape[1:3] != (height, width):
                raise ValueError(f"Mask size {mask_np.shape[1:3]} does not match image size {(height, width)}")
            alpha_np = mask_np
        elif has_alpha_channel:
            alpha_np = images_np[..., 3]
        else:
            alpha_np = np.ones((batch, height, width), dtype=np.float32)

        return [Image.fromarray((frame * 255).clip(0, 255).astype(np.uint8), mode="L") for frame in alpha_np]

    def _pixelate_batch(
        self,
        images: list[Image.Image],
        alpha_images: list[Image.Image],
        min_size: int,
        num_colors: int,
        init_mode: str,
        dither: bool,
        random_state: int = 42,
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        original_sizes = [image.size for image in images]

        downscaled = []
        downscaled_alpha = []
        for image, alpha_image, (width, height) in zip(images, alpha_images, original_sizes):
            if max(width, height) > min_size:
                if width > height:
                    new_width, new_height = min_size, int(height * (min_size / width))
                else:
                    new_width, new_height = int(width * (min_size / height)), min_size
                downscaled.append(image.resize((new_width, new_height), Image.NEAREST))
                downscaled_alpha.append(alpha_image.resize((new_width, new_height), Image.NEAREST))
            else:
                downscaled.append(image)
                downscaled_alpha.append(alpha_image)

        if init_mode != "none":
            downscaled = [
                self._quantize_colors(image, num_colors, init_mode, random_state, alpha)
                for image, alpha in zip(downscaled, downscaled_alpha)
            ]
        if dither:
            downscaled = [self._floyd_steinberg_dither(image, num_colors) for image in downscaled]

        pixelated = [image.resize(size, Image.NEAREST) for image, size in zip(downscaled, original_sizes)]
        pixelated_alpha = [
            alpha.resize(size, Image.NEAREST) for alpha, size in zip(downscaled_alpha, original_sizes)
        ]
        return pixelated, pixelated_alpha

    @classmethod
    def _quantize_colors(
        cls,
        image: Image.Image,
        num_colors: int,
        init_mode: str,
        random_state: int,
        alpha: Image.Image,
    ) -> Image.Image:
        np_image = np.array(image)
        pixels = np_image.reshape(-1, 3)

        # Fit the palette only on opaque pixels so transparent background
        # colors don't skew the cluster centers.
        alpha_np = np.array(alpha, dtype=np.float32).reshape(-1) / 255.0
        opaque = alpha_np > cls._ALPHA_FIT_THRESHOLD
        fit_pixels = pixels[opaque] if opaque.any() else pixels

        kmeans = KMeans(
            n_clusters=min(num_colors, len(fit_pixels)),
            init=init_mode,
            max_iter=cls._MAX_ITERATIONS,
            tol=1e-3,
            random_state=random_state,
            n_init="auto",
        )
        kmeans.fit(fit_pixels)
        colors = kmeans.cluster_centers_.astype(np.uint8)

        # Only quantize opaque pixels; transparent ones are blacked out
        # instead of being assigned a nearest fitted color, so the
        # background never soaks up one of the fitted palette colors.
        quantized = np.zeros_like(pixels)
        if opaque.any():
            quantized[opaque] = colors[kmeans.predict(pixels[opaque])]
        else:
            quantized = colors[kmeans.predict(pixels)]

        return Image.fromarray(quantized.reshape(np_image.shape))

    @staticmethod
    def _floyd_steinberg_dither(image: Image.Image, num_colors: int) -> Image.Image:
        arr = np.array(image, dtype=float) / 255
        width, height = image.size

        for y in range(height):
            for x in range(width):
                old_val = arr[y, x].copy()
                new_val = np.round(old_val * (num_colors - 1)) / (num_colors - 1)
                arr[y, x] = new_val
                err = old_val - new_val

                if x < width - 1:
                    arr[y, x + 1] += err * 7 / 16
                if y < height - 1:
                    if x > 0:
                        arr[y + 1, x - 1] += err * 3 / 16
                    arr[y + 1, x] += err * 5 / 16
                    if x < width - 1:
                        arr[y + 1, x + 1] += err / 16

        return Image.fromarray(np.array(arr * 255, dtype=np.uint8))
