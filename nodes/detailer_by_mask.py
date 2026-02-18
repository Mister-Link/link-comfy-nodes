"""Video detailer — mirrors Impact Pack's enhance_detail approach for video."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.samplers
import nodes


class VideoDetailer:
    """
    Video equivalent of FaceDetailer, but details the whole frame (or masked
    region) instead of detecting faces.

    For each frame batch:
      1. Crop to the masked bounding box (or full frame if no mask)
      2. Upscale the crop so its shortest side = guide_size
      3. VAE encode → ksampler → VAE decode
      4. Downscale result back to crop size
      5. Blend back into the original frame using the feathered mask
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide_size": (
                    "INT",
                    {"default": 512, "min": 64, "max": 8192, "step": 8},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01},
                ),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "noise_mask_feather": (
                    "INT",
                    {"default": 20, "min": 0, "max": 100, "step": 1},
                ),
                "basic_pipe": ("BASIC_PIPE",),
            },
            "optional": {
                "image_frames": ("IMAGE",),
                "latent": ("LATENT",),
                "mask_opt": ("MASK",),
                "ref_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Video detailer — upscales the masked region to guide_size, samples, "
        "then downscales and blends back. Like FaceDetailer but for the whole frame."
    )

    @staticmethod
    def _gaussian_blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Blur a [H,W] or [B,H,W] mask for feathering."""
        if radius <= 0:
            return mask

        kernel_size = radius * 2 + 1
        min_dim = min(mask.shape[-1], mask.shape[-2])
        if min_dim <= kernel_size:
            kernel_size = min_dim // 2
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
        k1d = gauss / gauss.sum()
        k2d = (k1d.unsqueeze(0) * k1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        pad = kernel_size // 2

        def _blur_one(m2d):
            return (
                F.conv2d(m2d.unsqueeze(0).unsqueeze(0), k2d, padding=pad)
                .squeeze(0)
                .squeeze(0)
            )

        if mask.ndim == 2:
            return _blur_one(mask)
        return torch.stack([_blur_one(m) for m in mask])

    @staticmethod
    def _fix_decoded_shape(decoded: torch.Tensor, expected_height: int) -> torch.Tensor:
        """Normalise WAN VAE output [B,F,d1,d2,C] → [F,H,W,C]."""
        if decoded.ndim == 5:
            b, f, d1, d2, c = decoded.shape
            decoded = decoded.reshape(b * f, d1, d2, c)
        if decoded.shape[1] != expected_height:
            decoded = torch.rot90(decoded, k=3, dims=(1, 2)).contiguous()
        return decoded

    @staticmethod
    def _mask_bbox(mask: torch.Tensor):
        """Return (x1,y1,x2,y2) bounding box of nonzero region across all frames."""
        combined = mask.max(dim=0)[0] if mask.ndim == 3 else mask
        nz = torch.nonzero(combined > 0.01)
        if len(nz) == 0:
            h, w = combined.shape
            return 0, 0, w, h
        y1 = int(nz[:, 0].min())
        y2 = int(nz[:, 0].max()) + 1
        x1 = int(nz[:, 1].min())
        x2 = int(nz[:, 1].max()) + 1
        return x1, y1, x2, y2

    @staticmethod
    def _snap(v: int, multiple: int) -> int:
        return ((v + multiple - 1) // multiple) * multiple

    def execute(
        self,
        guide_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        feather,
        noise_mask_feather,
        basic_pipe,
        image_frames=None,
        latent=None,
        mask_opt=None,
        ref_image=None,
    ):
        if image_frames is None and latent is None:
            raise ValueError(
                "[Video Detailer] Either image_frames or latent must be provided."
            )

        model, clip, vae, positive, negative = basic_pipe

        # ── LATENT PATH ─────────────────────────────────────────────────────────
        # Pass the full latent; noise_mask restricts denoising to the masked area.
        if image_frames is None:
            latent_samples = latent["samples"]  # [B, C, F, H, W]
            num_frames = latent_samples.shape[2]
            img_height = latent_samples.shape[3] * 8
            img_width = latent_samples.shape[4] * 8

            if mask_opt is None:
                mask = torch.ones(
                    (num_frames, img_height, img_width), dtype=torch.float32
                )
            else:
                mask = mask_opt
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0).expand(num_frames, -1, -1)
                elif mask.shape[0] != num_frames:
                    mask = mask[0:1].expand(num_frames, -1, -1)

            feathered = self._gaussian_blur_mask(mask, noise_mask_feather)
            noise_mask = feathered.unsqueeze(1).to(latent_samples.device)  # [F,1,H,W]

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

            decoded = vae.decode(samples["samples"])
            decoded = self._fix_decoded_shape(decoded, img_height)
            output_mask = mask[0] if mask.ndim == 3 else mask
            return (decoded, output_mask)

        # ── IMAGE PATH ──────────────────────────────────────────────────────────
        # Mirrors Impact Pack enhance_detail:
        #   crop bbox → upscale to guide_size → encode → sample → decode
        #   → downscale → paste back with feathered mask
        num_frames = image_frames.shape[0]
        img_height = image_frames.shape[1]
        img_width = image_frames.shape[2]

        if mask_opt is None:
            mask = torch.ones((num_frames, img_height, img_width), dtype=torch.float32)
        else:
            mask = mask_opt
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(num_frames, -1, -1)
            elif mask.shape[0] != num_frames:
                mask = mask[0:1].expand(num_frames, -1, -1)

        # 1. Get bounding box of the masked region
        x1, y1, x2, y2 = self._mask_bbox(mask)
        # Snap to 8-pixel boundary so VAE is happy
        x1 = max(0, (x1 // 8) * 8)
        y1 = max(0, (y1 // 8) * 8)
        x2 = min(img_width, self._snap(x2, 8))
        y2 = min(img_height, self._snap(y2, 8))
        crop_w = x2 - x1
        crop_h = y2 - y1
        print(f"[Video Detailer] Crop bbox: ({x1},{y1})→({x2},{y2})  {crop_w}×{crop_h}")

        # 2. Crop frames and mask to bbox
        crop_frames = image_frames[:, y1:y2, x1:x2, :]  # [F, crop_h, crop_w, C]
        crop_mask = mask[:, y1:y2, x1:x2]  # [F, crop_h, crop_w]

        # 3. Upscale so shortest side = guide_size, keep divisible by 64
        scale = guide_size / min(crop_h, crop_w)
        up_h = self._snap(int(crop_h * scale), 64)
        up_w = self._snap(int(crop_w * scale), 64)
        print(f"[Video Detailer] Upscale: {crop_w}×{crop_h} → {up_w}×{up_h}")

        up_frames = F.interpolate(
            crop_frames.permute(0, 3, 1, 2),  # [F,C,crop_h,crop_w]
            size=(up_h, up_w),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)  # [F, up_h, up_w, C]

        # 4. Build noise mask at upscaled size
        up_mask = F.interpolate(
            crop_mask.unsqueeze(1).float(),  # [F,1,crop_h,crop_w]
            size=(up_h, up_w),
            mode="bilinear",
            align_corners=False,
        )  # [F, 1, up_h, up_w]
        if noise_mask_feather > 0:
            up_mask = self._gaussian_blur_mask(
                up_mask.squeeze(1), noise_mask_feather
            ).unsqueeze(1)

        # 5. Encode → sample → decode
        # If ref_image provided, use it as the latent starting point so the
        # sampler denoises from that content rather than from the input frames.
        if ref_image is not None:
            # Resize ref_image to match the upscaled crop size, tiled across frames
            ref_up = (
                F.interpolate(
                    ref_image[:1, :, :, :3].permute(0, 3, 1, 2),
                    size=(up_h, up_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .expand(num_frames, -1, -1, -1)
            )
            encoded = vae.encode(ref_up)
            print(f"[Video Detailer] Using ref_image as latent base {ref_up.shape}")
        else:
            encoded = vae.encode(up_frames[:, :, :, :3])  # [F, C, lH, lW]
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

        decoded = vae.decode(samples["samples"])  # [F, up_h, up_w, C]
        decoded = self._fix_decoded_shape(decoded, up_h)

        # 6. Downscale back to crop size
        enhanced = F.interpolate(
            decoded.permute(0, 3, 1, 2),  # [F,C,up_h,up_w]
            size=(crop_h, crop_w),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)  # [F, crop_h, crop_w, C]

        # 7. Paste back into output using feathered mask
        output_frames = image_frames.clone()
        for fi in range(num_frames):
            fm = crop_mask[fi].clone()  # [crop_h, crop_w]
            if feather > 0:
                fm = self._gaussian_blur_mask(fm, feather)
            fm = fm.unsqueeze(-1)  # [crop_h, crop_w, 1]
            orig = output_frames[fi, y1:y2, x1:x2, :]
            enh = enhanced[fi].to(orig.device)
            output_frames[fi, y1:y2, x1:x2, :] = orig * (1 - fm) + enh * fm

        output_mask = mask[0] if mask.ndim == 3 else mask
        print(f"[Video Detailer] Done — {output_frames.shape}")
        return (output_frames, output_mask)


__all__ = ["VideoDetailer"]
