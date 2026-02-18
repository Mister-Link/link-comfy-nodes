"""Video detailer — per-frame crop→denoise→paste inspired by Impact MaskDetailer."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import nodes
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion


class VideoDetailer:
    """
    MaskDetailer-style node for video.

    For each frame in the decoded video:
      1. Crop the masked region (with padding via crop_factor)
      2. Upscale the crop to guide_size (LANCZOS)
      3. VAE encode → KSampler img2img at low denoise → VAE decode
      4. Downscale back to original crop size (LANCZOS)
      5. Feathered alpha-blend paste onto the original frame

    Because denoise < 1.0, the KSampler starts from the encoded crop itself
    (img2img), so output naturally stays close to the input.
    Unmasked areas are never touched — they come from the original decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "basic_pipe": ("BASIC_PIPE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0001, "max": 1.0, "step": 0.01},
                ),
                "feather": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Gaussian blur radius for the paste-back mask edge.",
                    },
                ),
                "noise_mask_feather": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Gaussian blur radius for the latent noise mask "
                        "(DifferentialDiffusion boundary softness).",
                    },
                ),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "How much to expand the crop beyond the mask bbox. "
                        "Larger = more context for the model.",
                    },
                ),
                "guide_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Target resolution for the short side of the crop "
                        "before denoising. Higher = finer detail.",
                    },
                ),
                "max_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Maximum resolution cap after upscaling.",
                    },
                ),
            },
            "optional": {
                "mask_opt": ("MASK",),
                "image_model_pipe": (
                    "BASIC_PIPE",
                    {
                        "tooltip": "Optional image (non-video) model pipe for "
                        "per-frame img2img. If not provided, falls back to "
                        "video-latent denoise + pixel composite."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "MaskDetailer for video. Crops masked regions per-frame, "
        "upscales, img2img denoises, downscales, and pastes back with "
        "feathered blending. Unmasked areas are never modified."
    )

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _gaussian_blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Gaussian blur a 2D or 3D (batch of 2D) mask tensor."""
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
    def _mask_bbox(mask_2d: torch.Tensor):
        """Get tight bounding box [x1, y1, x2, y2] of nonzero region in a 2D mask."""
        rows = torch.any(mask_2d > 0.5, dim=1)
        cols = torch.any(mask_2d > 0.5, dim=0)
        if not rows.any():
            return None
        y1, y2 = torch.where(rows)[0][[0, -1]]
        x1, x2 = torch.where(cols)[0][[0, -1]]
        return (x1.item(), y1.item(), x2.item() + 1, y2.item() + 1)

    @staticmethod
    def _expand_crop_region(bbox, crop_factor, img_w, img_h):
        """Expand bbox by crop_factor, clamped to image bounds."""
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_w = bw * crop_factor
        new_h = bh * crop_factor
        nx1 = max(0, int(cx - new_w / 2))
        ny1 = max(0, int(cy - new_h / 2))
        nx2 = min(img_w, int(cx + new_w / 2))
        ny2 = min(img_h, int(cy + new_h / 2))
        return (nx1, ny1, nx2, ny2)

    @staticmethod
    def _resize_tensor_image(image: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Resize [1,H,W,C] image tensor using bilinear interpolation."""
        # [1,H,W,C] → [1,C,H,W]
        img = image.permute(0, 3, 1, 2)
        img = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
        # [1,C,H,W] → [1,H,W,C]
        return img.permute(0, 2, 3, 1)

    @staticmethod
    def _scale_vace_strength(conditioning, mult):
        """Deep-copy conditioning and multiply all vace_strength values."""
        if mult == 1.0:
            return conditioning
        out = []
        for tensor, meta in conditioning:
            meta = copy.copy(meta)
            if "vace_strength" in meta:
                meta["vace_strength"] = [s * mult for s in meta["vace_strength"]]
            out.append([tensor, meta])
        return out

    # ---------------------------------------------- per-frame detail pipeline

    def _detail_frame(
        self,
        frame: torch.Tensor,
        mask_2d: torch.Tensor,
        model,
        vae,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        feather,
        noise_mask_feather,
        crop_factor,
        guide_size,
        max_size,
    ) -> torch.Tensor:
        """Crop→upscale→img2img→downscale→paste for one frame.

        Args:
            frame: [H, W, C] single frame
            mask_2d: [H, W] mask for this frame
        Returns:
            [H, W, C] refined frame
        """
        img_h, img_w = frame.shape[0], frame.shape[1]

        # Find mask bbox
        bbox = self._mask_bbox(mask_2d)
        if bbox is None:
            return frame  # empty mask, nothing to do

        # Expand crop region
        crop_region = self._expand_crop_region(bbox, crop_factor, img_w, img_h)
        cx1, cy1, cx2, cy2 = crop_region
        crop_w, crop_h = cx2 - cx1, cy2 - cy1

        # Crop image and mask
        cropped_image = frame[cy1:cy2, cx1:cx2, :].unsqueeze(0)  # [1, ch, cw, C]
        cropped_mask = mask_2d[cy1:cy2, cx1:cx2]  # [ch, cw]

        # Compute upscale
        bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        upscale = guide_size / max(min(bbox_w, bbox_h), 1)
        new_w = int(crop_w * upscale)
        new_h = int(crop_h * upscale)
        if new_w > max_size or new_h > max_size:
            upscale *= max_size / max(new_w, new_h)
            new_w = int(crop_w * upscale)
            new_h = int(crop_h * upscale)
        # Round to nearest 8 for VAE
        new_w = max(8, (new_w // 8) * 8)
        new_h = max(8, (new_h // 8) * 8)

        if upscale < 1.0:
            upscale = 1.0
            new_w = max(8, (crop_w // 8) * 8)
            new_h = max(8, (crop_h // 8) * 8)

        # Upscale crop
        upscaled = self._resize_tensor_image(cropped_image, new_w, new_h)

        # Build noise mask for the upscaled region
        upscaled_mask = (
            F.interpolate(
                cropped_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        # Feather the noise mask for DifferentialDiffusion
        if noise_mask_feather > 0:
            upscaled_mask = self._gaussian_blur_mask(upscaled_mask, noise_mask_feather)

        # VAE encode
        latent = vae.encode(upscaled.squeeze(0).unsqueeze(0))  # expects [B, H, W, C]
        latent_dict = {"samples": latent}

        # Attach noise mask if not a full mask
        if upscaled_mask.mean() < 0.99:
            # Apply DifferentialDiffusion
            detail_model = model
            if "denoise_mask_function" not in detail_model.model_options:
                detail_model = DifferentialDiffusion.execute(detail_model)[0]
            noise_mask_latent = upscaled_mask.unsqueeze(0).unsqueeze(0)
            latent_dict["noise_mask"] = noise_mask_latent

            samples = nodes.common_ksampler(
                detail_model,
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
        else:
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
        refined = vae.decode(samples["samples"])
        # Handle shape — image VAE returns [B, H, W, C]
        if refined.ndim == 5:
            refined = refined.squeeze(0)
        if refined.ndim == 4:
            refined = refined[0]  # [H, W, C]

        # Downscale back to original crop size
        refined = self._resize_tensor_image(
            refined.unsqueeze(0), crop_w, crop_h
        ).squeeze(0)  # [crop_h, crop_w, C]

        # Build feathered paste mask
        paste_mask = self._gaussian_blur_mask(cropped_mask, feather)
        paste_mask = paste_mask.unsqueeze(-1).to(frame.device)  # [ch, cw, 1]

        # Composite: paste refined crop back onto frame
        output = frame.clone()
        region = output[cy1:cy2, cx1:cx2, :]
        output[cy1:cy2, cx1:cx2, :] = (
            1 - paste_mask
        ) * region + paste_mask * refined.to(frame.device)

        return output

    # ---------------------------------------------- video-latent fallback

    def _detail_video_latent(
        self,
        latent_samples,
        original_frames,
        mask,
        num_frames,
        model,
        vae,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        feather,
        noise_mask_feather,
        img_height,
    ) -> torch.Tensor:
        """Fallback: denoise the full video latent + pixel composite."""
        feathered_latent = self._gaussian_blur_mask(mask, noise_mask_feather)
        noise_mask = feathered_latent.unsqueeze(1).to(latent_samples.device)

        if "denoise_mask_function" not in model.model_options:
            model = DifferentialDiffusion.execute(model)[0]

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

        enhanced_decoded = vae.decode(samples["samples"])
        enhanced_frames = self._fix_decoded_shape(enhanced_decoded, img_height)

        composite_mask = self._gaussian_blur_mask(mask, feather)
        mask_4d = composite_mask.unsqueeze(-1).to(original_frames.device)

        return (1 - mask_4d) * original_frames + mask_4d * enhanced_frames

    # ---------------------------------------------- main entry point

    def execute(
        self,
        latent,
        basic_pipe,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        feather,
        noise_mask_feather,
        crop_factor,
        guide_size,
        max_size,
        mask_opt=None,
        image_model_pipe=None,
    ):
        video_model, clip, video_vae, positive, negative = basic_pipe

        # WAN latent: [B, C, F, H, W]
        latent_samples = latent["samples"]
        img_height = latent_samples.shape[3] * 8
        img_width = latent_samples.shape[4] * 8

        print(
            f"[Video Detailer] latent {latent_samples.shape} → {img_width}×{img_height} px"
        )

        # Decode the original video latent to pixel frames
        original_decoded = video_vae.decode(latent_samples)
        original_frames = self._fix_decoded_shape(original_decoded, img_height)
        num_frames = original_frames.shape[0]
        print(f"[Video Detailer] decoded {num_frames} frames {original_frames.shape}")

        # Build pixel-space mask [F, H, W]
        if mask_opt is None:
            mask = torch.ones((num_frames, img_height, img_width), dtype=torch.float32)
        else:
            mask = mask_opt.clone()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(num_frames, -1, -1).contiguous()
            elif mask.shape[0] != num_frames:
                mask = mask[0:1].expand(num_frames, -1, -1).contiguous()

        # Choose pipeline based on whether an image model is available
        if image_model_pipe is not None:
            # Per-frame crop→denoise→paste using the image model
            img_model, img_clip, img_vae, img_pos, img_neg = image_model_pipe
            print(
                f"[Video Detailer] per-frame detail with image model, {num_frames} frames"
            )

            output_frames = []
            for i in range(num_frames):
                frame = original_frames[i]  # [H, W, C]
                frame_mask = mask[i]  # [H, W]

                refined = self._detail_frame(
                    frame,
                    frame_mask,
                    img_model,
                    img_vae,
                    img_pos,
                    img_neg,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    denoise,
                    feather,
                    noise_mask_feather,
                    crop_factor,
                    guide_size,
                    max_size,
                )
                output_frames.append(refined)
                print(f"[Video Detailer] frame {i + 1}/{num_frames} done")

            output = torch.stack(output_frames)
        else:
            # Fallback: denoise full video latent + pixel composite
            print(
                "[Video Detailer] no image_model_pipe — video latent denoise + composite"
            )
            output = self._detail_video_latent(
                latent_samples,
                original_frames,
                mask,
                num_frames,
                video_model,
                video_vae,
                positive,
                negative,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                denoise,
                feather,
                noise_mask_feather,
                img_height,
            )

        print(f"[Video Detailer] output {output.shape}")
        output_mask = mask[0] if mask.ndim == 3 else mask
        return (output, output_mask)


__all__ = ["VideoDetailer"]
