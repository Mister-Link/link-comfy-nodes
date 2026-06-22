from __future__ import annotations

import gc

import torch
import torch.nn.functional as F

import comfy.model_base
import comfy.model_management
import comfy.samplers
import nodes


class VACESampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_frames": ("IMAGE",),
                "mask": ("MASK",),
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
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "1.0 = fully regenerate masked regions.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "WAN/VACE masked video inpainter. Flanks the masked span with the nearest "
        "unmasked frames as temporal anchors, builds VACE conditioning from "
        "image_frames and mask, samples from an all-zero latent (standard VACE "
        "approach), then composites the result back over the originals."
    )

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
                    f"mask has {mask.shape[0]} frames but video has {frame_count}."
                )
        else:
            raise ValueError("mask must be 2D or 3D.")

        return mask.clamp_(0.0, 1.0)

    @staticmethod
    def _resize_phantom_in_conditioning(
        conditioning: list,
        latent_height: int,
        latent_width: int,
        drop: bool = False,
    ) -> list:
        """Resize Phantom time_dim_concat metadata to match the sampled latent."""
        result = []
        for tensor, meta in conditioning:
            t = meta.get("time_dim_concat")
            if not isinstance(t, torch.Tensor):
                result.append((tensor, meta))
                continue
            if drop:
                new_meta = dict(meta)
                del new_meta["time_dim_concat"]
                result.append((tensor, new_meta))
                continue
            if t.shape[-2] != latent_height or t.shape[-1] != latent_width:
                b, c, frames, h, w = t.shape
                resized = F.interpolate(
                    t.contiguous().reshape(b * c * frames, 1, h, w),
                    size=(latent_height, latent_width),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, frames, latent_height, latent_width)
                new_meta = dict(meta)
                new_meta["time_dim_concat"] = resized
                result.append((tensor, new_meta))
            else:
                result.append((tensor, meta))
        return result

    @staticmethod
    def _build_vace_conditioning(
        vae,
        frames: torch.Tensor,
        mask: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_count, height, width = mask.shape
        vae_stride = 8
        latent_height = height // vae_stride
        latent_width = width // vae_stride

        mask_3ch = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        centered = frames - 0.5
        inactive = (centered * (1.0 - mask_3ch) + 0.5).clamp(0.0, 1.0)
        reactive = (centered * mask_3ch + 0.5).clamp(0.0, 1.0)

        inactive_lat = vae.encode(inactive[:, :, :, :3]).to(device)
        reactive_lat = vae.encode(reactive[:, :, :, :3]).to(device)
        vace_frames = torch.cat([inactive_lat, reactive_lat], dim=1)
        latent_frames = vace_frames.shape[2]

        packed_mask = mask.view(
            frame_count, latent_height, vae_stride, latent_width, vae_stride
        )
        packed_mask = packed_mask.permute(2, 4, 0, 1, 3).reshape(
            vae_stride * vae_stride, frame_count, latent_height, latent_width
        ).float()  # [64, T, lH, lW]

        # WAN-aware temporal grouping: latent 0 = pixel frame 0 only,
        # latent k = max over pixel frames (k-1)*4+1 to k*4.
        # nearest-exact interpolation picks one representative per bin and
        # silently misses masked frames whose representative is unmasked,
        # so the model receives no reactive VACE signal for those latents.
        n_ch = vae_stride * vae_stride
        vace_m = packed_mask.new_zeros(n_ch, latent_frames, latent_height, latent_width)
        vace_m[:, 0] = packed_mask[:, 0]
        for k in range(1, latent_frames):
            start = (k - 1) * 4 + 1
            end = min(start + 4, frame_count)
            if start < frame_count:
                vace_m[:, k] = packed_mask[:, start:end].amax(dim=1)

        return vace_frames, vace_m.unsqueeze(0).to(device)

    @staticmethod
    def _fix_decoded_frames(decoded: torch.Tensor, expected_frames: int) -> torch.Tensor:
        if decoded.ndim == 5:
            decoded = decoded.squeeze(0)
        if decoded.shape[0] > expected_frames:
            decoded = decoded[:expected_frames]
        return decoded

    @staticmethod
    def _find_mask_spans(mask: torch.Tensor) -> list[tuple[int, int]]:
        masked_frames = mask.amax(dim=(1, 2)) > 0.0
        spans: list[tuple[int, int]] = []
        idx = 0
        while idx < int(mask.shape[0]):
            if not bool(masked_frames[idx]):
                idx += 1
                continue
            start = idx
            while idx + 1 < int(mask.shape[0]) and bool(masked_frames[idx + 1]):
                idx += 1
            spans.append((start, idx))
            idx += 1
        return spans

    @staticmethod
    def _get_anchor_augmentation(
        frames: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Flanks the masked span with the nearest unmasked frames as anchors.

        Prev anchor is prepended, next anchor is appended — preserving temporal
        order [prev, frames…, next] so the model can interpolate across the gap.
        """
        spans = VACESampler._find_mask_spans(mask)
        if not spans:
            return frames, mask, 0, 0

        prepend_f: list[torch.Tensor] = []
        prepend_m: list[torch.Tensor] = []
        append_f: list[torch.Tensor] = []
        append_m: list[torch.Tensor] = []

        first_start, _ = spans[0]
        prev_idx = first_start - 1
        while prev_idx >= 0 and bool(mask[prev_idx].amax() > 0.0):
            prev_idx -= 1
        if prev_idx >= 0:
            prepend_f.append(frames[prev_idx:prev_idx + 1])
            prepend_m.append(torch.zeros_like(mask[prev_idx:prev_idx + 1]))

        _, last_end = spans[-1]
        next_idx = last_end + 1
        while next_idx < int(mask.shape[0]) and bool(mask[next_idx].amax() > 0.0):
            next_idx += 1
        if next_idx < int(mask.shape[0]):
            append_f.append(frames[next_idx:next_idx + 1])
            append_m.append(torch.zeros_like(mask[next_idx:next_idx + 1]))

        if not prepend_f and not append_f:
            return frames, mask, 0, 0

        aug_frames = torch.cat(prepend_f + [frames] + append_f, dim=0)
        aug_mask = torch.cat(prepend_m + [mask] + append_m, dim=0)
        return aug_frames, aug_mask, len(prepend_f), len(append_f)

    def _run_pass(
        self,
        frames: torch.Tensor,
        mask: torch.Tensor,
        model,
        vae,
        positive,
        negative,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
    ) -> torch.Tensor:
        device = frames.device
        aug_frames, aug_mask, trim_prefix, trim_suffix = self._get_anchor_augmentation(
            frames, mask
        )

        frame_count = int(aug_frames.shape[0])
        height = int(aug_frames.shape[1])
        width = int(aug_frames.shape[2])
        latent_height = height // 8
        latent_width = width // 8
        latent_frames = ((frame_count - 1) // 4) + 1

        import node_helpers

        vace_frames, vace_mask = self._build_vace_conditioning(
            vae, aug_frames, aug_mask, device
        )
        vace_values = {
            "vace_frames": [vace_frames],
            "vace_mask": [vace_mask],
            "vace_strength": [1.0],
        }
        positive_cond = node_helpers.conditioning_set_values(positive, vace_values)
        negative_cond = node_helpers.conditioning_set_values(negative, vace_values)

        patch_emb = getattr(
            getattr(model.model, "diffusion_model", None), "patch_embedding", None
        )
        drop_time_dim = (
            patch_emb is not None
            and hasattr(patch_emb, "weight")
            and patch_emb.weight.shape[1] > 16
        )
        positive_cond = self._resize_phantom_in_conditioning(
            positive_cond, latent_height, latent_width, drop=drop_time_dim
        )
        negative_cond = self._resize_phantom_in_conditioning(
            negative_cond, latent_height, latent_width, drop=drop_time_dim
        )

        # All-zero latent: VACE conditioning drives the generation entirely.
        # Encoding fill_source into the latent and using a noise_mask was
        # fighting the VACE module — the standard WanVaceToVideo workflow
        # always starts from zeros and lets inactive/reactive channels guide output.
        latent = torch.zeros(
            [1, 16, latent_frames, latent_height, latent_width],
            device=comfy.model_management.intermediate_device(),
        )
        start_step = int(steps * (1.0 - denoise))

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
            {"samples": latent},
            start_step,
            steps,
            "disable",
        )[0]

        refined = vae.decode(sampled["samples"]).to(device=device, dtype=torch.float32)
        refined = self._fix_decoded_frames(refined, aug_frames.shape[0]).clamp_(0.0, 1.0)
        if refined.shape[0] < aug_frames.shape[0]:
            refined = torch.cat([refined, aug_frames[refined.shape[0]:]], dim=0)
        end_idx = refined.shape[0] - trim_suffix if trim_suffix > 0 else refined.shape[0]
        refined = refined[trim_prefix:end_idx]

        output = frames * (1.0 - mask.unsqueeze(-1)) + refined * mask.unsqueeze(-1)
        return output.clamp_(0.0, 1.0)

    def sample(
        self,
        image_frames,
        mask,
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
    ):
        device = comfy.model_management.get_torch_device()
        frames = image_frames.to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
        frame_count = int(frames.shape[0])
        height = int(frames.shape[1])
        width = int(frames.shape[2])

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"VACE Sampler expects frame dimensions divisible by 16; got {width}x{height}. "
                "Resize upstream if needed."
            )
        if frame_count < 5:
            raise ValueError(f"image_frames has {frame_count} frame(s); minimum is 5.")
        if isinstance(model.model, comfy.model_base.WAN21) and (frame_count - 1) % 4 != 0:
            rem = (frame_count - 1) % 4
            lower = frame_count - rem
            upper = lower + 4
            raise ValueError(
                f"{frame_count} frames is not valid for WAN (must satisfy 1 + n×4). "
                f"Nearest valid counts: {lower} or {upper}."
            )
        if not isinstance(model.model, comfy.model_base.WAN21_Vace):
            raise ValueError(
                "VACE Sampler requires a WAN VACE model. Load a VACE-merged model and "
                "connect normal positive/negative conditioning."
            )

        prepared_mask = self._prepare_mask(mask, frame_count, height, width, device)
        if prepared_mask.amax() < 1e-6:
            return (frames.cpu(),)

        output = self._run_pass(
            frames, prepared_mask, model, vae, positive, negative,
            seed, steps, cfg, sampler_name, scheduler, denoise,
        )

        comfy.model_management.soft_empty_cache()
        gc.collect()
        return (output.clamp(0.0, 1.0).cpu(),)
