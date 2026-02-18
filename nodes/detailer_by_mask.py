"""Video detailer with dimension fixing and batch processing."""

from __future__ import annotations

import torch

import comfy.samplers
import nodes


class VideoDetailer:
    """
    Detailer for video frames with NAG compatibility.

    Splits the image into a subdivide×subdivide grid of tiles. Each tile is
    upscaled so its shortest side equals guide_size, sampled, decoded, then
    downscaled back to the original tile size and blended into the output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide_size": (
                    "INT",
                    {"default": 720, "min": 64, "max": 8192, "step": 8},
                ),
                "subdivide": ("INT", {"default": 2, "min": 1, "max": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm"}),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"default": "simple"},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0001, "max": 1.0, "step": 0.01},
                ),
                "feather": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "basic_pipe": ("BASIC_PIPE",),
            },
            "optional": {
                "image_frames": ("IMAGE",),
                "latent": ("LATENT",),
                "mask_opt": ("MASK",),
                "noise_mask_feather": (
                    "INT",
                    {"default": 10, "min": 0, "max": 100, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Video detailer — splits image into subdivide×subdivide tiles, "
        "upscales each to guide_size, samples, then downscales back and blends."
    )

    @staticmethod
    def _round_to(value: int, multiple: int) -> int:
        """Round up to nearest multiple."""
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _fix_decoded_shape(decoded: torch.Tensor, expected_height: int) -> torch.Tensor:
        """Normalise WAN VAE output to [F, H, W, C].

        WAN VAE outputs content rotated 90° CCW in [B, F, d1, d2, C] form.
        After reshape to [F, d1, d2, C], if d1 != expected_height the spatial
        dims and pixel content both need correcting via rot90(k=3) (90° CW).
        """
        if decoded.ndim == 5:
            b, f, d1, d2, c = decoded.shape
            decoded = decoded.reshape(b * f, d1, d2, c)
        if decoded.shape[1] != expected_height:
            decoded = torch.rot90(decoded, k=3, dims=(1, 2)).contiguous()
        return decoded

    @staticmethod
    def _gaussian_blur_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply gaussian blur to mask for feathering."""
        if kernel_size <= 0:
            return mask

        kernel_size = kernel_size * 2 + 1

        if mask.shape[-1] <= kernel_size or mask.shape[-2] <= kernel_size:
            kernel_size = min(mask.shape[-1], mask.shape[-2]) // 2
            if kernel_size % 2 == 0:
                kernel_size -= 1
            if kernel_size < 3:
                return mask

        sigma = kernel_size / 3.0
        x = (
            torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
            - kernel_size // 2
        )
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        padding = kernel_size // 2
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            blurred = torch.nn.functional.conv2d(mask, kernel_2d, padding=padding)
            return blurred.squeeze(0).squeeze(0)
        elif mask.ndim == 3:
            result = []
            for m in mask:
                m = m.unsqueeze(0).unsqueeze(0)
                blurred = torch.nn.functional.conv2d(m, kernel_2d, padding=padding)
                result.append(blurred.squeeze(0).squeeze(0))
            return torch.stack(result)
        return mask

    def execute(
        self,
        guide_size,
        subdivide,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        feather,
        basic_pipe,
        image_frames=None,
        latent=None,
        mask_opt=None,
        noise_mask_feather=10,
    ):
        if image_frames is None and latent is None:
            raise ValueError(
                "[Video Detailer] Either image_frames or latent must be provided."
            )

        divisor = 64  # required for NAG / WAN VAE

        if image_frames is not None:
            num_frames = image_frames.shape[0]
            img_height = image_frames.shape[1]
            img_width = image_frames.shape[2]
        else:
            # WAN latent: [B, C, F, H, W]
            latent_samples = latent["samples"]
            num_frames = latent_samples.shape[2]
            img_height = latent_samples.shape[3] * 8
            img_width = latent_samples.shape[4] * 8

        # Build mask [F, H, W]
        if mask_opt is None:
            mask = torch.ones((num_frames, img_height, img_width), dtype=torch.float32)
        else:
            mask = mask_opt
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(num_frames, -1, -1)
            elif mask.ndim == 3 and mask.shape[0] != num_frames:
                mask = mask[0:1].expand(num_frames, -1, -1)

        print(
            f"[Video Detailer] {num_frames} frames, {img_width}x{img_height}, "
            f"subdivide={subdivide}, guide_size={guide_size}"
        )

        model, clip, vae, positive, negative = basic_pipe

        # ── IMAGE PATH ──────────────────────────────────────────────────────────
        if image_frames is not None:
            output_frames = image_frames.clone()

            # Divide into subdivide×subdivide equal tiles.
            # Last tile in each axis absorbs any remainder pixels.
            tile_h = img_height // subdivide
            tile_w = img_width // subdivide

            for row in range(subdivide):
                ty1 = row * tile_h
                ty2 = ty1 + tile_h if row < subdivide - 1 else img_height

                for col in range(subdivide):
                    tx1 = col * tile_w
                    tx2 = tx1 + tile_w if col < subdivide - 1 else img_width

                    t_h = ty2 - ty1
                    t_w = tx2 - tx1

                    # Skip tiles with no mask coverage
                    tile_mask = mask[:, ty1:ty2, tx1:tx2]  # [F, t_h, t_w]
                    if tile_mask.max() < 0.01:
                        print(f"[Video Detailer] Tile ({col},{row}) skipped — no mask")
                        continue

                    # Crop tile frames: [F, t_h, t_w, C]
                    tile_frames = image_frames[:, ty1:ty2, tx1:tx2, :]

                    # Upscale tile so shortest side = guide_size, divisible by 64
                    scale = guide_size / min(t_h, t_w)
                    up_h = self._round_to(int(t_h * scale), divisor)
                    up_w = self._round_to(int(t_w * scale), divisor)
                    print(
                        f"[Video Detailer] Tile ({col},{row}): "
                        f"{t_w}x{t_h} → {up_w}x{up_h} → {t_w}x{t_h}"
                    )

                    # [F, t_h, t_w, C] → [F, C, t_h, t_w] → interpolate → [F, C, up_h, up_w]
                    up_frames = torch.nn.functional.interpolate(
                        tile_frames.permute(0, 3, 1, 2),
                        size=(up_h, up_w),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # → [F, up_h, up_w, C]

                    # Encode all frames together: vae.encode expects [F, H, W, C]
                    # and returns a standard image latent [F, C, lH, lW]
                    encoded = vae.encode(up_frames[:, :, :, :3])  # [F, C, lH, lW]

                    # Upscale tile mask to match upscaled tile size
                    up_mask = torch.nn.functional.interpolate(
                        tile_mask.unsqueeze(1).float(),  # [F, 1, t_h, t_w]
                        size=(up_h, up_w),
                        mode="bilinear",
                        align_corners=False,
                    )  # [F, 1, up_h, up_w]
                    if noise_mask_feather > 0:
                        up_mask = self._gaussian_blur_mask(
                            up_mask.squeeze(1), noise_mask_feather
                        ).unsqueeze(1)

                    latent_dict = {"samples": encoded, "noise_mask": up_mask}

                    samples = nodes.common_ksampler(
                        model,
                        seed,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        positive,
                        negative,
                        latent_dict,
                        denoise=denoise,
                    )[0]

                    # Decode: returns [F, up_h, up_w, C] (image VAE, no rotation needed)
                    decoded = vae.decode(samples["samples"])
                    decoded = self._fix_decoded_shape(decoded, up_h)
                    # decoded: [F, up_h, up_w, C]

                    # Downscale back to original tile size
                    enhanced = torch.nn.functional.interpolate(
                        decoded.permute(0, 3, 1, 2),  # [F, C, up_h, up_w]
                        size=(t_h, t_w),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [F, t_h, t_w, C]

                    # Blend per-frame using feathered mask
                    for fi in range(num_frames):
                        fm = tile_mask[fi].clone()  # [t_h, t_w]
                        if feather > 0:
                            fm = self._gaussian_blur_mask(fm, feather)
                        fm = fm.unsqueeze(-1).to(output_frames.device)  # [t_h, t_w, 1]
                        orig = output_frames[fi, ty1:ty2, tx1:tx2, :]
                        enh = enhanced[fi].to(output_frames.device)
                        output_frames[fi, ty1:ty2, tx1:tx2, :] = (
                            orig * (1 - fm) + enh * fm
                        )

            output_mask = mask[0] if mask.ndim == 3 else mask
            print(f"[Video Detailer] Done — output {output_frames.shape}")
            return (output_frames, output_mask)

        # ── LATENT PATH ─────────────────────────────────────────────────────────
        # VACE/WAN: latent must stay full-size; noise_mask restricts denoising.
        expected_decode_height = img_height
        print(
            f"[Video Detailer] Latent path — shape {latent_samples.shape}, "
            f"{img_width}x{img_height}"
        )

        feathered_mask = (
            self._gaussian_blur_mask(mask, noise_mask_feather)
            if noise_mask_feather > 0
            else mask
        )
        noise_mask = feathered_mask.unsqueeze(1).to(
            latent_samples.device
        )  # [F, 1, H, W]

        latent_dict = {"samples": latent_samples, "noise_mask": noise_mask}

        samples = nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_dict,
            denoise=denoise,
        )[0]

        decoded_frames = vae.decode(samples["samples"])
        print(f"[Video Detailer] Decoded {decoded_frames.shape}")
        decoded_frames = self._fix_decoded_shape(decoded_frames, expected_decode_height)
        print(f"[Video Detailer] Fixed  {decoded_frames.shape}")

        output_mask = mask[0] if mask.ndim == 3 else mask
        return (decoded_frames, output_mask)


__all__ = ["VideoDetailer"]
