from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.latent_formats
import comfy.model_management
import comfy.samplers
import node_helpers
import nodes
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion


class VideoDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "basic_pipe": ("BASIC_PIPE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "feather": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Gaussian blur radius for pixel-space paste-back.",
                    },
                ),
                "noise_mask_feather": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Gaussian blur for the ref/frame boundary in "
                        "latent noise mask.",
                    },
                ),
            },
            "optional": {
                "mask_opt": ("MASK",),
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image stitched to the left of each "
                        "frame. Protected from denoising — the model sees "
                        "it as context and keeps output consistent."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Reference-guided video detailer. Stitches reference beside each "
        "frame, denoises only the frame half, then crops and composites back."
    )

    @staticmethod
    def _gaussian_blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
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
        if decoded.ndim == 5:
            b, f, d1, d2, c = decoded.shape
            decoded = decoded.reshape(b * f, d1, d2, c)
        if decoded.shape[1] != expected_height:
            decoded = torch.rot90(decoded, k=3, dims=(1, 2)).contiguous()
        return decoded

    @staticmethod
    def _build_vace_conditioning(
        composite_video,
        denoise_mask_pixel,
        vae,
        latent_frames,
        img_height,
        width_double,
        strength,
    ):
        num_pixel_frames = composite_video.shape[0]

        greyed_video = (
            composite_video[:, :, :, :3] * (1 - denoise_mask_pixel)
            + 0.5 * denoise_mask_pixel
        )

        ones_mask = torch.ones(
            (num_pixel_frames, img_height, width_double, 1),
            dtype=torch.float32,
        )

        control = greyed_video - 0.5
        inactive = (control * (1 - ones_mask[:, :, :, :1])) + 0.5
        reactive = (control * ones_mask[:, :, :, :1]) + 0.5

        inactive_latent = vae.encode(inactive[:, :, :, :3])
        reactive_latent = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive_latent, reactive_latent), dim=1)

        print(f"[Video Detailer] VACE latent {control_video_latent.shape}")

        vae_stride = 8
        height_mask = img_height // vae_stride
        width_mask = width_double // vae_stride

        mask_2d = ones_mask[:, :, :, 0]
        mask_blocks = mask_2d.view(
            num_pixel_frames, height_mask, vae_stride, width_mask, vae_stride
        )
        mask_blocks = mask_blocks.permute(2, 4, 0, 1, 3)
        mask_blocks = mask_blocks.reshape(
            vae_stride * vae_stride, num_pixel_frames, height_mask, width_mask
        )
        mask_blocks = F.interpolate(
            mask_blocks.unsqueeze(0),
            size=(latent_frames, height_mask, width_mask),
            mode="nearest-exact",
        ).squeeze(0)

        vace_mask = mask_blocks.unsqueeze(0)

        print(
            f"[Video Detailer] VACE mask {vace_mask.shape} mean={vace_mask.mean():.3f}"
        )

        return [control_video_latent], [vace_mask]

    def execute(
        self,
        latent,
        basic_pipe,
        seed,
        steps,
        start_step,
        end_step,
        cfg,
        sampler_name,
        scheduler,
        feather,
        noise_mask_feather,
        mask_opt=None,
        reference_image=None,
    ):
        model, clip, vae, positive, negative = basic_pipe

        latent_samples = latent["samples"]
        latent_frames = latent_samples.shape[2]
        img_height = latent_samples.shape[3] * 8
        img_width = latent_samples.shape[4] * 8

        print(
            f"[Video Detailer] latent {latent_samples.shape} "
            f"-> {img_width}x{img_height} px"
        )

        original_decoded = vae.decode(latent_samples)
        original_frames = self._fix_decoded_shape(original_decoded, img_height)
        num_pixel_frames = original_frames.shape[0]
        print(
            f"[Video Detailer] decoded {num_pixel_frames} frames "
            f"{original_frames.shape}"
        )

        if mask_opt is None:
            mask = torch.ones(
                (num_pixel_frames, img_height, img_width), dtype=torch.float32
            )
        else:
            mask = mask_opt.clone()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(num_pixel_frames, -1, -1).contiguous()
            elif mask.shape[0] != num_pixel_frames:
                mask = mask[0:1].expand(num_pixel_frames, -1, -1).contiguous()

        if reference_image is not None:
            ref = reference_image[0] if reference_image.ndim == 4 else reference_image
            if ref.shape[0] != img_height or ref.shape[1] != img_width:
                ref = (
                    F.interpolate(
                        ref.unsqueeze(0).permute(0, 3, 1, 2),
                        size=(img_height, img_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                )
            print(f"[Video Detailer] reference {ref.shape}")
        else:
            ref = original_frames[0]
            print("[Video Detailer] using first frame as reference")

        half_w = ((img_width + 15) // 16) * 16
        width_double = half_w * 2

        ref_resized = (
            F.interpolate(
                ref.unsqueeze(0).permute(0, 3, 1, 2),
                size=(img_height, half_w),
                mode="bilinear",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .squeeze(0)
        )

        composite_list = []
        for i in range(num_pixel_frames):
            frame = original_frames[i]
            frame_resized = (
                F.interpolate(
                    frame.unsqueeze(0).permute(0, 3, 1, 2),
                    size=(img_height, half_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .squeeze(0)
            )
            composite_list.append(torch.cat([ref_resized, frame_resized], dim=1))

        composite_video = torch.stack(composite_list)
        print(f"[Video Detailer] composite {composite_video.shape}")

        composite_latent = vae.encode(composite_video[:, :, :, :3])
        print(f"[Video Detailer] encoded composite {composite_latent.shape}")

        denoise_mask_pixel = torch.zeros(
            (num_pixel_frames, img_height, width_double, 1), dtype=torch.float32
        )
        for i in range(num_pixel_frames):
            fm = mask[i]
            if fm.shape[1] != half_w:
                fm = (
                    F.interpolate(
                        fm.unsqueeze(0).unsqueeze(0),
                        size=(img_height, half_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            denoise_mask_pixel[i, :, half_w:, 0] = fm

        vace_strength = 1.0
        if len(positive) > 0 and len(positive[0]) > 1:
            existing_strengths = positive[0][1].get("vace_strength", None)
            if existing_strengths is not None and len(existing_strengths) > 0:
                vace_strength = existing_strengths[0]
                print(f"[Video Detailer] inheriting VACE strength={vace_strength}")

        vace_frames_list, vace_mask_list = self._build_vace_conditioning(
            composite_video, denoise_mask_pixel, vae, latent_frames,
            img_height, width_double, vace_strength,
        )

        vace_values = {
            "vace_frames": vace_frames_list,
            "vace_mask": vace_mask_list,
            "vace_strength": [vace_strength],
        }
        positive = node_helpers.conditioning_set_values(positive, vace_values)
        negative = node_helpers.conditioning_set_values(negative, vace_values)
        print("[Video Detailer] VACE conditioning rebuilt for double-width")

        noise_mask_wide = torch.zeros(
            (num_pixel_frames, img_height, width_double), dtype=torch.float32
        )
        for i in range(num_pixel_frames):
            fm = mask[i]
            if fm.shape[1] != half_w:
                fm = (
                    F.interpolate(
                        fm.unsqueeze(0).unsqueeze(0),
                        size=(img_height, half_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            noise_mask_wide[i, :, half_w:] = fm

        noise_mask_wide = self._gaussian_blur_mask(noise_mask_wide, noise_mask_feather)
        noise_mask_4d = noise_mask_wide.unsqueeze(1).to(composite_latent.device)

        print(
            f"[Video Detailer] noise_mask {noise_mask_4d.shape} "
            f"mean={noise_mask_4d.mean():.3f}"
        )

        if "denoise_mask_function" not in model.model_options:
            model = DifferentialDiffusion.execute(model)[0]
            print("[Video Detailer] DifferentialDiffusion applied")

        latent_dict = {
            "samples": composite_latent,
            "noise_mask": noise_mask_4d,
        }

        samples = nodes.NODE_CLASS_MAPPINGS["KSamplerAdvanced"]().sample(
            model, "enable", seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_dict, start_step, end_step, "disable",
        )[0]

        decoded_wide = vae.decode(samples["samples"])
        decoded_wide = self._fix_decoded_shape(decoded_wide, img_height)
        print(f"[Video Detailer] decoded wide {decoded_wide.shape}")

        refined_half = decoded_wide[:, :, half_w:, :]

        if refined_half.shape[2] != img_width:
            rf = refined_half.permute(0, 3, 1, 2)
            rf = F.interpolate(
                rf, size=(img_height, img_width), mode="bilinear", align_corners=False,
            )
            refined_half = rf.permute(0, 2, 3, 1)

        print(f"[Video Detailer] refined frames {refined_half.shape}")

        composite_mask = self._gaussian_blur_mask(mask, feather)
        mask_4d = composite_mask.unsqueeze(-1).to(original_frames.device)

        out_n = min(original_frames.shape[0], refined_half.shape[0])
        output = (1 - mask_4d[:out_n]) * original_frames[:out_n] + mask_4d[
            :out_n
        ] * refined_half[:out_n].to(original_frames.device)
        print(f"[Video Detailer] output {output.shape}")

        output_mask = mask[0] if mask.ndim == 3 else mask
        return (output, output_mask)
