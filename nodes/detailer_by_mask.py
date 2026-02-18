"""Video detailer with dimension fixing and batch processing."""

from __future__ import annotations

import torch

import comfy.samplers
import nodes


class VideoDetailer:
    """
    Detailer for video frames with NAG compatibility.

    This node:
    - Optionally accepts a MASK to detail specific regions
    - If no mask provided, details the entire image
    - Fixes crop regions to be divisible by 64 (required for NAG)
    - Processes all frames together as a batch
    - Does not depend on Impact Pack for sampling
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide_size": (
                    "INT",
                    {"default": 720, "min": 64, "max": 8192, "step": 8},
                ),
                "columns": ("INT", {"default": 2, "min": 1, "max": 32}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 32}),
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
                "trim_latent": ("INT", {"default": 0, "min": 0, "max": 1024}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Video detailer with NAG compatibility - processes all frames as a batch"
    )

    @staticmethod
    def _fix_to_divisor(value: int, divisor: int) -> int:
        """Round up to nearest multiple of divisor."""
        return ((value + divisor - 1) // divisor) * divisor

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
            # rot90(k=3) on spatial dims: corrects 90° CCW rotation in content
            # and swaps [F, W, H, C] → [F, H, W, C]
            decoded = torch.rot90(decoded, k=3, dims=(1, 2)).contiguous()
        return decoded

    @staticmethod
    def _gaussian_blur_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply gaussian blur to mask for feathering."""
        if kernel_size <= 0:
            return mask

        # Ensure odd kernel size
        kernel_size = kernel_size * 2 + 1

        # Check if mask is too small for kernel
        if mask.shape[-1] <= kernel_size or mask.shape[-2] <= kernel_size:
            kernel_size = min(mask.shape[-1], mask.shape[-2]) // 2
            if kernel_size % 2 == 0:
                kernel_size -= 1
            if kernel_size < 3:
                return mask

        # Create gaussian kernel
        sigma = kernel_size / 3.0
        x = (
            torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
            - kernel_size // 2
        )
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        # Apply blur
        padding = kernel_size // 2
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            blurred = torch.nn.functional.conv2d(mask, kernel_2d, padding=padding)
            return blurred.squeeze(0).squeeze(0)
        elif mask.ndim == 3:
            # [B, H, W] -> blur each
            result = []
            for m in mask:
                m = m.unsqueeze(0).unsqueeze(0)
                blurred = torch.nn.functional.conv2d(m, kernel_2d, padding=padding)
                result.append(blurred.squeeze(0).squeeze(0))
            return torch.stack(result)
        else:
            return mask

    def execute(
        self,
        guide_size,
        columns,
        rows,
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
        trim_latent=0,
    ):
        """Process all frames as a batch."""

        if image_frames is None and latent is None:
            raise ValueError(
                "[Video Detailer] Either image_frames or latent must be provided."
            )

        divisor = 64  # NAG divisor

        # Determine dimensions from whichever input is provided
        if image_frames is not None:
            num_frames = image_frames.shape[0]
            img_height = image_frames.shape[1]
            img_width = image_frames.shape[2]
        else:
            # WAN latent is [B, C, F, H, W] — shape[0]=B, shape[2]=F, shape[3]=H, shape[4]=W
            latent_samples = latent["samples"]
            # Trim reference frames prepended by WanVaceToVideo when reference_image is used
            if trim_latent > 0:
                latent_samples = latent_samples[:, :, trim_latent:, :, :]
                print(
                    f"[Video Detailer] Trimmed {trim_latent} reference frames, latent now: {latent_samples.shape}"
                )
            num_frames = latent_samples.shape[2]
            img_height = latent_samples.shape[3] * 8
            img_width = latent_samples.shape[4] * 8

        # If no mask provided, create a full mask
        if mask_opt is None:
            print(f"[Video Detailer] No mask provided, using full image")
            mask = torch.ones((num_frames, img_height, img_width), dtype=torch.float32)
        else:
            mask = mask_opt

        print(f"[Video Detailer] Processing {num_frames} frames as batch")

        # Handle mask dimensions
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(num_frames, -1, -1)
        elif mask.ndim == 3 and mask.shape[0] != num_frames:
            if mask.shape[0] == 1:
                mask = mask.expand(num_frames, -1, -1)
            else:
                mask = mask[0:1].expand(num_frames, -1, -1)

        print(f"[Video Detailer] Mask shape: {mask.shape}")

        # Get model components from basic_pipe
        model, clip, vae, positive, negative = basic_pipe

        if image_frames is not None:
            # --- IMAGE PATH: tile grid → upscale each tile → encode → sample → decode → paste ---
            output_frames = image_frames.clone()

            tile_h = img_height // rows
            tile_w = img_width // columns
            print(
                f"[Video Detailer] Tiling {columns}x{rows} grid, tile size {tile_w}x{tile_h}"
            )

            for row in range(rows):
                for col in range(columns):
                    ty1 = row * tile_h
                    ty2 = ty1 + tile_h if row < rows - 1 else img_height
                    tx1 = col * tile_w
                    tx2 = tx1 + tile_w if col < columns - 1 else img_width

                    t_h = ty2 - ty1
                    t_w = tx2 - tx1

                    # Skip tile if mask has no coverage
                    tile_mask = mask[:, ty1:ty2, tx1:tx2]
                    if tile_mask.max() < 0.01:
                        print(
                            f"[Video Detailer] Tile ({col},{row}) skipped — no mask coverage"
                        )
                        continue

                    tile_frames = image_frames[:, ty1:ty2, tx1:tx2, :]

                    # Upscale tile to guide_size on shortest side
                    scale = guide_size / min(t_h, t_w)
                    new_h = self._fix_to_divisor(int(t_h * scale), divisor)
                    new_w = self._fix_to_divisor(int(t_w * scale), divisor)
                    print(
                        f"[Video Detailer] Tile ({col},{row}): {t_w}x{t_h} -> {new_w}x{new_h}"
                    )

                    tile_nchw = tile_frames.permute(0, 3, 1, 2)
                    upscaled = torch.nn.functional.interpolate(
                        tile_nchw,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)

                    upscaled_mask = torch.nn.functional.interpolate(
                        tile_mask.unsqueeze(1),
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    if noise_mask_feather > 0:
                        upscaled_mask = self._gaussian_blur_mask(
                            upscaled_mask, noise_mask_feather
                        )

                    latent_samples = vae.encode(upscaled[:, :, :, :3])
                    latent_dict = {
                        "samples": latent_samples,
                        "noise_mask": upscaled_mask.unsqueeze(1),
                    }

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

                    decoded = vae.decode(samples["samples"])
                    decoded = self._fix_decoded_shape(decoded, new_h)

                    # Downscale back to tile size
                    decoded_nchw = decoded.permute(0, 3, 1, 2)
                    enhanced = torch.nn.functional.interpolate(
                        decoded_nchw,
                        size=(t_h, t_w),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)

                    # Blend using feathered tile mask
                    for frame_idx in range(num_frames):
                        fm = tile_mask[frame_idx].clone()
                        if feather > 0:
                            fm = self._gaussian_blur_mask(fm, feather)
                        fm = fm.unsqueeze(-1).to(output_frames.device)
                        enh = enhanced[frame_idx].to(output_frames.device)
                        orig = output_frames[frame_idx, ty1:ty2, tx1:tx2, :]
                        output_frames[frame_idx, ty1:ty2, tx1:tx2, :] = (
                            orig * (1 - fm) + enh * fm
                        )

            print(f"[Video Detailer] Done - output shape: {output_frames.shape}")
            output_mask = mask[0] if mask.ndim == 3 else mask
            return (output_frames, output_mask)
        else:
            # --- LATENT PATH: pass full latent unchanged; mask restricts denoising ---
            # VACE/WAN models require the latent shape to exactly match the conditioning
            # context, so we must never crop/resize the latent tensor itself.
            expected_decode_height = img_height
            print(f"[Video Detailer] Latent input — passing full latent to sampler")
            # latent_samples was already set (and trimmed if needed) above
            print(
                f"[Video Detailer] Latent shape: {latent_samples.shape}, img_height={img_height}, img_width={img_width}"
            )

            # Build a full pixel-space noise mask [F, 1, H, W].
            # ComfyUI's reshape_mask resizes it to latent space internally.
            feathered_mask = mask
            if noise_mask_feather > 0:
                feathered_mask = self._gaussian_blur_mask(mask, noise_mask_feather)
            noise_mask = feathered_mask.unsqueeze(1).to(latent_samples.device)

            print(
                f"[Video Detailer] noise_mask shape: {noise_mask.shape}, min: {noise_mask.min():.3f}, max: {noise_mask.max():.3f}, mean: {noise_mask.mean():.3f}"
            )

            latent_dict = {
                "samples": latent_samples,
                "noise_mask": noise_mask,
            }

        # Sample using ComfyUI's native ksampler
        print(
            f"[Video Detailer] Sampling with {sampler_name}/{scheduler}, {steps} steps, denoise={denoise}..."
        )

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

        # VAE decode
        print(f"[Video Detailer] VAE decoding...")
        decoded_frames = vae.decode(samples["samples"])
        print(f"[Video Detailer] Decoded shape: {decoded_frames.shape}")
        decoded_frames = self._fix_decoded_shape(decoded_frames, expected_decode_height)
        print(f"[Video Detailer] After shape fix: {decoded_frames.shape}")

        # LATENT PATH: the noise mask already restricted denoising to the target
        # region during sampling, so decoded_frames is the final result directly.
        decoded_num_frames = decoded_frames.shape[0]
        print(
            f"[Video Detailer] Latent path: {num_frames} latent frames -> {decoded_num_frames} decoded frames"
        )
        output_frames = decoded_frames

        print(f"[Video Detailer] Done - output shape: {output_frames.shape}")

        # Return output — always return a single [H, W] mask from the first frame
        output_mask = mask[0] if mask.ndim == 3 else mask
        return (output_frames, output_mask)


__all__ = ["VideoDetailer"]
