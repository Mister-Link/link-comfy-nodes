from __future__ import annotations

import gc

import torch
import torch.nn.functional as F

import comfy.model_base
import comfy.model_management
import comfy.samplers
import node_helpers
import nodes
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel


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
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How much to re-denoise. 0 = no change, 1 = full re-generation.",
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
                "upscale_model": (
                    "UPSCALE_MODEL",
                    {
                        "tooltip": (
                            "Optional upscale model (e.g. ESRGAN) applied to frames "
                            "before encoding. Produces higher-quality detail than "
                            "bilinear upscaling. Frames are downscaled back after sampling."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"

    DESCRIPTION = (
        "Wan/VACE video detailer for blurry or noisy frame batches. Uses the source "
        "video as VACE control guidance during inpainting-style refinement."
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
    def _slice_controlnet_hint(control, start_frame: int, end_frame: int):
        """Recursively clone a ControlNet and slice its hint to [start_frame, end_frame)."""
        c = control.copy()
        if c.cond_hint_original is not None:
            hint = c.cond_hint_original
            if hint.shape[0] > 1:
                c.cond_hint_original = hint[start_frame : min(end_frame, hint.shape[0])]
        if control.previous_controlnet is not None:
            c.previous_controlnet = VideoDetailer._slice_controlnet_hint(
                control.previous_controlnet, start_frame, end_frame
            )
        return c

    @staticmethod
    def _slice_conditioning_for_chunk(
        conditioning: list, start_frame: int, end_frame: int
    ) -> list:
        """Slice ControlNet hints in conditioning to cover only [start_frame, end_frame)."""
        result = []
        for tensor, meta in conditioning:
            if "control" not in meta:
                result.append((tensor, meta))
                continue
            new_meta = dict(meta)
            new_meta["control"] = VideoDetailer._slice_controlnet_hint(
                meta["control"], start_frame, end_frame
            )
            result.append((tensor, new_meta))
        return result

    @staticmethod
    def _extract_phantom_latent(conditioning: list) -> torch.Tensor | None:
        for _, meta in conditioning:
            t = meta.get("time_dim_concat")
            if isinstance(t, torch.Tensor):
                return t
        return None

    @staticmethod
    def _extract_upstream_reference_latent(
        conditioning: list,
    ) -> torch.Tensor | None:
        for _, meta in conditioning:
            vace_frames = meta.get("vace_frames")
            vace_masks = meta.get("vace_mask")
            if not isinstance(vace_frames, list) or not isinstance(vace_masks, list):
                continue
            if not vace_frames or not vace_masks:
                continue

            frames = vace_frames[0]
            mask = vace_masks[0]
            if not isinstance(frames, torch.Tensor) or not isinstance(
                mask, torch.Tensor
            ):
                continue
            if frames.ndim != 5 or mask.ndim != 5:
                continue

            total_frames = min(frames.shape[2], mask.shape[2])
            reference_frames = 0
            for index in range(total_frames):
                if mask[:, :, index].abs().max() < 1e-6:
                    reference_frames += 1
                else:
                    break

            if reference_frames > 0:
                return frames[:, :, :reference_frames].clone()

        return None

    @staticmethod
    def _resize_phantom_in_conditioning(
        conditioning: list, latent_height: int, latent_width: int
    ) -> list:
        result = []
        for tensor, meta in conditioning:
            t = meta.get("time_dim_concat")
            if isinstance(t, torch.Tensor) and (
                t.shape[-2] != latent_height or t.shape[-1] != latent_width
            ):
                b, c, frames, h, w = t.shape
                resized = F.interpolate(
                    t.view(b * c * frames, 1, h, w),
                    size=(latent_height, latent_width),
                    mode="bilinear",
                    align_corners=False,
                ).view(b, c, frames, latent_height, latent_width)
                new_meta = dict(meta)
                new_meta["time_dim_concat"] = resized
                result.append((tensor, new_meta))
            else:
                result.append((tensor, meta))
        return result

    @staticmethod
    def _resize_reference_latent(
        reference_latent: torch.Tensor,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        if (
            reference_latent.shape[-2] == latent_height
            and reference_latent.shape[-1] == latent_width
        ):
            return reference_latent

        b, c, t, h, w = reference_latent.shape
        return F.interpolate(
            reference_latent.view(b * c * t, 1, h, w),
            size=(latent_height, latent_width),
            mode="bilinear",
            align_corners=False,
        ).view(b, c, t, latent_height, latent_width)

    @staticmethod
    def _upscale_frames_with_model(
        upscale_model,
        images: torch.Tensor,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        """Upscale frames through an upscale model then resize to exact target dims."""
        upscaled = ImageUpscaleWithModel().upscale(upscale_model, images)[0]
        upscaled = upscaled.to(device=images.device, dtype=torch.float32)
        if upscaled.shape[1] != target_height or upscaled.shape[2] != target_width:
            upscaled = VideoDetailer._resize_images(
                upscaled, target_height, target_width
            )
        return upscaled

    @staticmethod
    def _resize_vace_in_conditioning(
        conditioning: list,
        latent_height: int,
        latent_width: int,
    ) -> list:
        """Resize vace_frames and vace_mask latents to match a new spatial resolution.

        Safe to call when latents are already the right size — returns conditioning
        unchanged in that case. Must be called per-chunk so latent_height/latent_width
        match what the sampler will receive.
        """
        needs_resize = False
        for _, meta in conditioning:
            for key in ("vace_frames", "vace_mask"):
                for entry in meta.get(key, []):
                    if isinstance(entry, torch.Tensor) and entry.ndim == 5:
                        if (
                            entry.shape[-2] != latent_height
                            or entry.shape[-1] != latent_width
                        ):
                            needs_resize = True
                            break
                if needs_resize:
                    break
            if needs_resize:
                break
        if not needs_resize:
            return conditioning

        result = []
        for tensor, meta in conditioning:
            if "vace_frames" not in meta and "vace_mask" not in meta:
                result.append((tensor, meta))
                continue
            new_meta = dict(meta)
            if "vace_frames" in meta:
                new_list = []
                for vf in meta["vace_frames"]:
                    if isinstance(vf, torch.Tensor) and vf.ndim == 5:
                        b, c, t, h, w = vf.shape
                        if h != latent_height or w != latent_width:
                            vf = (
                                F.interpolate(
                                    vf.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w),
                                    size=(latent_height, latent_width),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                .reshape(b, t, c, latent_height, latent_width)
                                .permute(0, 2, 1, 3, 4)
                            )
                    new_list.append(vf)
                new_meta["vace_frames"] = new_list
            if "vace_mask" in meta:
                new_list = []
                for vm in meta["vace_mask"]:
                    if isinstance(vm, torch.Tensor) and vm.ndim == 5:
                        b, c, t, h, w = vm.shape
                        if h != latent_height or w != latent_width:
                            vm = (
                                F.interpolate(
                                    vm.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w),
                                    size=(latent_height, latent_width),
                                    mode="nearest",
                                )
                                .reshape(b, t, c, latent_height, latent_width)
                                .permute(0, 2, 1, 3, 4)
                            )
                    new_list.append(vm)
                new_meta["vace_mask"] = new_list
            result.append((tensor, new_meta))
        return result

    @staticmethod
    def _set_vace_conditioning(
        conditioning: list,
        vace_frames: torch.Tensor,
        vace_mask: torch.Tensor,
    ) -> list:
        # Append detailer's inpainting VACE stream to the existing conditioning.
        # append=True concatenates the lists, so any upstream VACE (e.g. pose
        # control from WanVaceToVideoControlStrength) is preserved alongside the
        # detailer's own inpainting context stream.
        return node_helpers.conditioning_set_values(
            conditioning,
            {
                "vace_frames": [vace_frames],
                "vace_mask": [vace_mask],
                "vace_strength": [1.0],
            },
            append=True,
        )

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
    def _build_vace_inputs(
        chunk_frames: torch.Tensor,
        chunk_masks: torch.Tensor,
        vae,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_length, height, width, _ = chunk_frames.shape
        latent_length = ((chunk_length - 1) // 4) + 1

        control_video = chunk_frames - 0.5
        mask = chunk_masks.unsqueeze(-1)
        inactive = (control_video * (1.0 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive_latent = vae.encode(inactive[:, :, :, :3])
        reactive_latent = vae.encode(reactive[:, :, :, :3])
        control_latent = torch.cat((inactive_latent, reactive_latent), dim=1)

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
        vace_mask = (
            F.adaptive_max_pool3d(
                mask_blocks.unsqueeze(0),
                output_size=(latent_length, latent_height, latent_width),
            )
            .squeeze(0)
            .unsqueeze(0)
        )

        return control_latent, vace_mask

    @staticmethod
    def _extend_vace_for_upstream_phantom(
        control_latent: torch.Tensor,
        vace_mask: torch.Tensor,
        phantom_latent_frames: int,
        latent_height: int,
        latent_width: int,
        target_height: int,
        target_width: int,
        vae,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if phantom_latent_frames <= 0:
            return control_latent, vace_mask

        # Match WanVaceAdvanced's native phantom handling: the phantom embed itself
        # stays in time_dim_concat, while vace_context only gets neutral control slots.
        phantom_frame_count = phantom_latent_frames * 4
        inactive_pixels = torch.full(
            (phantom_frame_count, target_height, target_width, 3),
            0.5,
            dtype=torch.float32,
            device=device,
        )
        reactive_pixels = torch.zeros_like(inactive_pixels)

        inactive_latent = vae.encode(inactive_pixels[:, :, :, :3]).to(device)
        reactive_latent = vae.encode(reactive_pixels[:, :, :, :3]).to(device)
        phantom_control_latent = torch.cat((inactive_latent, reactive_latent), dim=1)

        if (
            phantom_control_latent.shape[-2] != latent_height
            or phantom_control_latent.shape[-1] != latent_width
        ):
            raise ValueError(
                "Upstream phantom padding produced mismatched latent dimensions."
            )

        phantom_mask = torch.ones(
            vace_mask.shape[0],
            vace_mask.shape[1],
            phantom_latent_frames,
            latent_height,
            latent_width,
            device=device,
            dtype=vace_mask.dtype,
        )

        return (
            torch.cat([control_latent, phantom_control_latent], dim=2),
            torch.cat([vace_mask, phantom_mask], dim=2),
        )

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
        guide_size,
        max_size,
        feather,
        noise_mask_feather,
        mask_opt=None,
        upscale_model=None,
    ):
        device = comfy.model_management.get_torch_device()
        frames = image_frames.to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
        frame_count, height, width, _ = frames.shape
        if frame_count == 0:
            raise ValueError("image_frames is empty.")

        if isinstance(model.model, comfy.model_base.WAN21) and (frame_count - 1) % 4 != 0:
            rem = (frame_count - 1) % 4
            lower = frame_count - rem
            upper = lower + 4
            raise ValueError(
                f"Video Detailer received {frame_count} frames, which is not valid for "
                f"this WAN model (frame count must satisfy 1 + n×4). "
                f"Nearest valid counts: {lower} or {upper}."
            )

        mask = self._prepare_mask(mask_opt, frame_count, height, width, device)
        upstream_reference_latent = self._extract_upstream_reference_latent(positive)
        has_upstream_vace = any("vace_frames" in meta for _, meta in positive)

        target_width, target_height = self._pick_target_size(
            height, width, guide_size, max_size
        )
        if upscale_model is not None:
            frames_target = self._upscale_frames_with_model(
                upscale_model, frames, target_height, target_width
            )
        else:
            frames_target = self._resize_images(frames, target_height, target_width)
        mask_target = self._resize_masks(mask, target_height, target_width).clamp_(
            0.0, 1.0
        )
        noise_mask_target = self._gaussian_blur_mask(
            mask_target, noise_mask_feather
        ).clamp_(0.0, 1.0)
        composite_mask = self._gaussian_blur_mask(mask, feather).clamp_(0.0, 1.0)

        if (
            noise_mask_feather > 0
            and "denoise_mask_function" not in model.model_options
        ):
            model = DifferentialDiffusion.execute(model)[0]

        start_step = int(steps * (1.0 - denoise))
        end_step = steps

        if composite_mask.amax() < 1e-6:
            return (frames.cpu(), mask.cpu())

        all_latent = vae.encode(frames_target[:, :, :, :3]).to(device)
        latent_height = all_latent.shape[-2]
        latent_width = all_latent.shape[-1]

        if has_upstream_vace:
            ref_latent_length = (
                upstream_reference_latent.shape[2]
                if upstream_reference_latent is not None
                else 0
            )
            if ref_latent_length > 0:
                ref_for_concat = upstream_reference_latent[:, : all_latent.shape[1]].to(
                    device=device, dtype=torch.float32
                )
                ref_for_concat = self._resize_reference_latent(
                    ref_for_concat, latent_height, latent_width
                )
                combined_latent = torch.cat((ref_for_concat, all_latent), dim=2)
            else:
                combined_latent = all_latent
            positive_cond = positive
            negative_cond = negative
        else:
            control_latent, vace_mask = self._build_vace_inputs(
                frames_target, mask_target, vae
            )
            control_latent = control_latent.to(device)
            vace_mask = vace_mask.to(device)
            ref_latent_length = 0

            phantom = self._extract_phantom_latent(positive)
            if phantom is not None:
                control_latent, vace_mask = self._extend_vace_for_upstream_phantom(
                    control_latent=control_latent,
                    vace_mask=vace_mask,
                    phantom_latent_frames=phantom.shape[2],
                    latent_height=latent_height,
                    latent_width=latent_width,
                    target_height=target_height,
                    target_width=target_width,
                    vae=vae,
                    device=device,
                )

            combined_latent = all_latent
            positive_cond = self._set_vace_conditioning(
                positive, control_latent, vace_mask
            )
            negative_cond = self._set_vace_conditioning(
                negative, control_latent, vace_mask
            )

        noise_mask = self._build_noise_mask(
            noise_mask_target,
            all_latent.shape[2],
            ref_latent_length,
            latent_height,
            latent_width,
        )
        positive_cond = self._slice_conditioning_for_chunk(
            positive_cond, 0, frame_count
        )
        negative_cond = self._slice_conditioning_for_chunk(
            negative_cond, 0, frame_count
        )
        positive_cond = self._resize_phantom_in_conditioning(
            positive_cond, latent_height, latent_width
        )
        negative_cond = self._resize_phantom_in_conditioning(
            negative_cond, latent_height, latent_width
        )
        positive_cond = self._resize_vace_in_conditioning(
            positive_cond, latent_height, latent_width
        )
        negative_cond = self._resize_vace_in_conditioning(
            negative_cond, latent_height, latent_width
        )

        sampled = nodes.NODE_CLASS_MAPPINGS["KSamplerAdvanced"]().sample(
            model,
            "enable",
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive_cond,
            negative_cond,
            {"samples": combined_latent, "noise_mask": noise_mask},
            start_step,
            end_step,
            "disable",
        )[0]

        refined_latent = sampled["samples"][:, :, ref_latent_length:, :, :]
        refined = vae.decode(refined_latent).to(device=device, dtype=torch.float32)
        if refined.ndim == 5:
            refined = refined.squeeze(0)
        refined = refined.clamp_(0.0, 1.0)

        if refined.shape[1] != height or refined.shape[2] != width:
            refined = self._resize_images(refined, height, width, mode="bicubic")

        # WAN's temporal VAE may decode fewer frames than were encoded (e.g. 55→53).
        # Pad with original frames so the output frame count always matches the input.
        n_refined = refined.shape[0]
        n_frames = frames.shape[0]
        if n_refined < n_frames:
            refined = torch.cat([refined, frames[n_refined:]], dim=0)
        elif n_refined > n_frames:
            refined = refined[:n_frames]

        composite_mask_4d = composite_mask.unsqueeze(-1)
        output = (1.0 - composite_mask_4d) * frames + composite_mask_4d * refined
        self._cleanup()
        return (output.clamp(0.0, 1.0).cpu(), mask.cpu())
