from __future__ import annotations

import gc

import torch
import torch.nn.functional as F

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
                "image_frames": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How strongly each chunk is re-denoised.",
                    },
                ),
                "chunk_size": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Frames processed together for temporal coherence.",
                    },
                ),
                "chunk_overlap": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "Frames shared between neighboring chunks.",
                    },
                ),
                "guide_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Target minimum side while detailing.",
                    },
                ),
                "max_size": (
                    "INT",
                    {
                        "default": 1536,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Caps the target detailing resolution.",
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Softens the final composite mask on output frames.",
                    },
                ),
                "noise_mask_feather": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Softens the latent inpaint mask before sampling.",
                    },
                ),
            },
            "optional": {
                "mask_opt": ("MASK",),
                "reference_image": (
                    "IMAGE",
                    {"tooltip": "Identity/style anchor applied to every chunk."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Wan/VACE video detailer for blurry or noisy frame batches. Refines in "
        "overlapping temporal chunks, uses the source video as control guidance, "
        "and keeps identity stable with a reference frame."
    )

    @staticmethod
    def _gaussian_blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask

        kernel_size = radius * 2 + 1
        min_dim = min(mask.shape[-2], mask.shape[-1])
        if min_dim <= kernel_size:
            kernel_size = min_dim // 2
            if kernel_size % 2 == 0:
                kernel_size -= 1
            if kernel_size < 3:
                return mask

        sigma = kernel_size / 3.0
        coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device) - (
            kernel_size // 2
        )
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).view(
            1, 1, kernel_size, kernel_size
        )
        pad = kernel_size // 2

        def blur_frame(frame: torch.Tensor) -> torch.Tensor:
            return F.conv2d(
                frame.view(1, 1, frame.shape[0], frame.shape[1]),
                kernel_2d,
                padding=pad,
            ).view(frame.shape[0], frame.shape[1])

        if mask.ndim == 2:
            return blur_frame(mask)
        return torch.stack([blur_frame(frame) for frame in mask], dim=0)

    @staticmethod
    def _resize_images(
        images: torch.Tensor,
        height: int,
        width: int,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        if images.shape[1] == height and images.shape[2] == width:
            return images
        return (
            F.interpolate(
                images.permute(0, 3, 1, 2),
                size=(height, width),
                mode=mode,
                align_corners=False if mode in {"bilinear", "bicubic"} else None,
            )
            .permute(0, 2, 3, 1)
            .contiguous()
        )

    @staticmethod
    def _resize_masks(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if mask.shape[-2] == height and mask.shape[-1] == width:
            return mask
        return F.interpolate(
            mask.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    @staticmethod
    def _prepare_mask(
        mask_opt: torch.Tensor | None,
        frame_count: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask_opt is None:
            return torch.ones(
                (frame_count, height, width), dtype=torch.float32, device=device
            )

        mask = mask_opt.to(device=device, dtype=torch.float32)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(frame_count, -1, -1).contiguous()
        elif mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask.expand(frame_count, -1, -1).contiguous()
            elif mask.shape[0] != frame_count:
                raise ValueError(
                    "mask_opt frame count must be 1 or match image_frames."
                )
        else:
            raise ValueError("mask_opt must be 2D or 3D.")

        return mask.clamp_(0.0, 1.0)

    @staticmethod
    def _conditioning_without_vace(conditioning: list) -> list:
        cleaned: list = []
        for tensor, meta in conditioning:
            new_meta = dict(meta)
            new_meta.pop("vace_frames", None)
            new_meta.pop("vace_mask", None)
            new_meta.pop("vace_strength", None)
            cleaned.append((tensor, new_meta))
        return cleaned

    @staticmethod
    def _set_vace_conditioning(
        conditioning: list,
        vace_frames: torch.Tensor,
        vace_mask: torch.Tensor,
    ) -> list:
        # Preserve vace_strength already embedded in the conditioning
        strength = None
        for _, meta in conditioning:
            if "vace_strength" in meta:
                strength = meta["vace_strength"]
                break
        base = VideoDetailer._conditioning_without_vace(conditioning)
        values: dict = {
            "vace_frames": [vace_frames],
            "vace_mask": [vace_mask],
        }
        if strength is not None:
            values["vace_strength"] = strength
        return node_helpers.conditioning_set_values(base, values, append=True)

    @staticmethod
    def _pick_target_size(
        height: int, width: int, guide_size: int, max_size: int
    ) -> tuple[int, int]:
        min_side = min(height, width)
        scale = guide_size / float(min_side) if min_side > 0 else 1.0
        new_width = width
        new_height = height
        if scale > 1.0:
            new_width = int(round(width * scale))
            new_height = int(round(height * scale))
        if max(new_width, new_height) > max_size:
            limit = max_size / float(max(new_width, new_height))
            new_width = int(round(new_width * limit))
            new_height = int(round(new_height * limit))

        new_width = max(16, (new_width // 16) * 16)
        new_height = max(16, (new_height // 16) * 16)
        return new_width, new_height

    @staticmethod
    def _make_temporal_chunks(
        frame_count: int, chunk_size: int, overlap: int
    ) -> list[tuple[int, int]]:
        if frame_count <= 0:
            return []

        chunk_size = max(1, chunk_size)
        overlap = max(0, min(overlap, chunk_size - 1))

        windows: list[tuple[int, int]] = []
        start = 0
        while start < frame_count:
            end = min(frame_count, start + chunk_size)
            windows.append((start, end))
            if end >= frame_count:
                break
            start = end - overlap
        return windows

    @staticmethod
    def _chunk_weights(
        length: int,
        overlap: int,
        is_first: bool,
        is_last: bool,
        device: torch.device,
    ) -> torch.Tensor:
        weights = torch.ones(length, dtype=torch.float32, device=device)
        usable_overlap = min(overlap, max(0, length - 1))
        if usable_overlap <= 0:
            return weights

        fade = torch.linspace(0.0, 1.0, usable_overlap + 2, device=device)[1:-1]
        if not is_first:
            weights[:usable_overlap] = fade
        if not is_last:
            weights[-usable_overlap:] = torch.flip(fade, dims=[0])
        return weights

    @staticmethod
    def _build_vace_inputs(
        chunk_frames: torch.Tensor,
        chunk_masks: torch.Tensor,
        reference_frame: torch.Tensor,
        vae,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        chunk_length, height, width, _ = chunk_frames.shape
        latent_length = ((chunk_length - 1) // 4) + 1

        reference_latent = vae.encode(reference_frame[:, :, :, :3])
        reference_latent = torch.cat(
            [reference_latent, torch.zeros_like(reference_latent)], dim=1
        )
        reference_latent_length = reference_latent.shape[2]

        control_video = chunk_frames - 0.5
        mask = chunk_masks.unsqueeze(-1)
        inactive = (control_video * (1.0 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive_latent = vae.encode(inactive[:, :, :, :3])
        reactive_latent = vae.encode(reactive[:, :, :, :3])
        control_latent = torch.cat((inactive_latent, reactive_latent), dim=1)
        control_latent = torch.cat((reference_latent, control_latent), dim=2)

        latent_height = inactive_latent.shape[-2]
        latent_width = inactive_latent.shape[-1]
        stride_h = height // latent_height
        stride_w = width // latent_width
        mask_blocks = mask.reshape(
            chunk_length,
            latent_height,
            stride_h,
            latent_width,
            stride_w,
            1,
        )
        mask_blocks = mask_blocks.permute(2, 4, 0, 1, 3, 5).reshape(
            stride_h * stride_w, chunk_length, latent_height, latent_width
        )
        vace_mask = F.adaptive_max_pool3d(
            mask_blocks.unsqueeze(0),
            output_size=(latent_length, latent_height, latent_width),
        ).squeeze(0)
        ref_mask = torch.zeros_like(vace_mask[:, :reference_latent_length])
        vace_mask = torch.cat((ref_mask, vace_mask), dim=1).unsqueeze(0)

        return control_latent, vace_mask, reference_latent_length

    @staticmethod
    def _build_noise_mask(
        mask_chunk: torch.Tensor,
        latent_frames: int,
        ref_latent_length: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        latent_mask = F.interpolate(
            mask_chunk.unsqueeze(1),
            size=(latent_height, latent_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        latent_mask = F.adaptive_max_pool3d(
            latent_mask.unsqueeze(0).unsqueeze(0),
            output_size=(latent_frames, latent_height, latent_width),
        )
        ref_mask = torch.zeros(
            1,
            1,
            ref_latent_length,
            latent_height,
            latent_width,
            device=mask_chunk.device,
            dtype=latent_mask.dtype,
        )
        return torch.cat((ref_mask, latent_mask), dim=2)

    @staticmethod
    def _cleanup():
        comfy.model_management.soft_empty_cache()
        gc.collect()

    def execute(
        self,
        image_frames,
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
        chunk_size,
        chunk_overlap,
        guide_size,
        max_size,
        feather,
        noise_mask_feather,
        mask_opt=None,
        reference_image=None,
    ):
        if isinstance(model, str) and model == "DUMMY":
            raise ValueError(
                "Video Detailer requires a real Wan/VACE model."
            )

        device = comfy.model_management.get_torch_device()
        frames = image_frames.to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
        frame_count, height, width, _ = frames.shape
        if frame_count == 0:
            raise ValueError("image_frames is empty.")

        mask = self._prepare_mask(mask_opt, frame_count, height, width, device)
        composite_mask = self._gaussian_blur_mask(mask, feather).clamp_(0.0, 1.0)
        source_reference = (
            reference_image[:1] if reference_image is not None else frames[:1]
        ).to(device=device, dtype=torch.float32)
        source_reference = self._resize_images(source_reference, height, width)

        target_width, target_height = self._pick_target_size(
            height, width, guide_size, max_size
        )
        frames_target = self._resize_images(frames, target_height, target_width)
        reference_target = self._resize_images(
            source_reference, target_height, target_width
        )
        mask_target = self._resize_masks(mask, target_height, target_width).clamp_(
            0.0, 1.0
        )
        noise_mask_target = self._gaussian_blur_mask(
            mask_target, noise_mask_feather
        ).clamp_(0.0, 1.0)

        if (
            noise_mask_feather > 0
            and "denoise_mask_function" not in model.model_options
        ):
            model = DifferentialDiffusion.execute(model)[0]

        output_sum = torch.zeros_like(frames)
        weight_sum = torch.zeros(
            (frame_count, 1, 1, 1), dtype=torch.float32, device=device
        )
        windows = self._make_temporal_chunks(frame_count, chunk_size, chunk_overlap)
        start_step = min(steps, int(round(steps * (1.0 - denoise))))

        for index, (start, end) in enumerate(windows):
            chunk_frames = frames_target[start:end]
            chunk_masks = mask_target[start:end]
            chunk_noise_mask = noise_mask_target[start:end]

            control_latent, vace_mask, ref_latent_length = self._build_vace_inputs(
                chunk_frames, chunk_masks, reference_target, vae
            )
            control_latent = control_latent.to(device)
            vace_mask = vace_mask.to(device)
            chunk_latent = vae.encode(chunk_frames[:, :, :, :3]).to(device)
            latent_height = chunk_latent.shape[-2]
            latent_width = chunk_latent.shape[-1]
            combined_latent = torch.cat(
                (
                    control_latent[:, : chunk_latent.shape[1], :ref_latent_length],
                    chunk_latent,
                ),
                dim=2,
            )
            noise_mask = self._build_noise_mask(
                chunk_noise_mask,
                chunk_latent.shape[2],
                ref_latent_length,
                latent_height,
                latent_width,
            )

            positive_chunk = self._set_vace_conditioning(
                positive, control_latent, vace_mask
            )
            negative_chunk = self._set_vace_conditioning(
                negative, control_latent, vace_mask
            )

            sampled = nodes.NODE_CLASS_MAPPINGS["KSamplerAdvanced"]().sample(
                model,
                "enable",
                seed + start,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive_chunk,
                negative_chunk,
                {"samples": combined_latent, "noise_mask": noise_mask},
                start_step,
                steps,
                "disable",
            )[0]

            refined_latent = sampled["samples"][:, :, ref_latent_length:, :, :]
            refined_chunk = vae.decode(refined_latent).to(
                device=device, dtype=torch.float32
            )
            # Video VAE decodes to (B, T, H, W, C); drop the batch dim
            if refined_chunk.ndim == 5:
                refined_chunk = refined_chunk.squeeze(0)
            refined_chunk = refined_chunk.clamp_(0.0, 1.0)
            if refined_chunk.shape[1] != height or refined_chunk.shape[2] != width:
                refined_chunk = self._resize_images(refined_chunk, height, width)

            chunk_original = frames[start:end]
            chunk_composite_mask = composite_mask[start:end].unsqueeze(-1)
            chunk_output = (
                1.0 - chunk_composite_mask
            ) * chunk_original + chunk_composite_mask * refined_chunk

            weights = self._chunk_weights(
                end - start,
                chunk_overlap,
                is_first=index == 0,
                is_last=index == (len(windows) - 1),
                device=device,
            ).view(-1, 1, 1, 1)
            output_sum[start:end] += chunk_output * weights
            weight_sum[start:end] += weights

            del (
                chunk_frames,
                chunk_masks,
                chunk_noise_mask,
                control_latent,
                vace_mask,
                chunk_latent,
                combined_latent,
                noise_mask,
                sampled,
                refined_latent,
                refined_chunk,
                chunk_output,
                weights,
                positive_chunk,
                negative_chunk,
            )
            self._cleanup()

        output = output_sum / weight_sum.clamp_min(1e-6)
        return (output.clamp(0.0, 1.0).cpu(), mask.cpu())
