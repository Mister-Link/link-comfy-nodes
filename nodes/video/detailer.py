from __future__ import annotations

import gc

import torch
import torch.nn.functional as F

import comfy.model_base
import comfy.model_management
import comfy.samplers
import nodes
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel


class VideoDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
                        "tooltip": (
                            "How much to regenerate masked frames. "
                            "1.0 = full regeneration from scratch (use for missing/wrong frames). "
                            "Lower values refine without changing structure."
                        ),
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Softens the composite mask when blending output back onto original frames.",
                    },
                ),
                "noise_mask_feather": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": (
                            "Softens the noise mask spatially within each frame. "
                            "0 = hard boundary (recommended for whole-frame temporal gaps). "
                            "Higher values blend the noise level at mask edges (useful for spatial inpainting)."
                        ),
                    },
                ),
            },
            "optional": {
                "image_frames": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Video frames to encode and inpaint. When provided, frames are "
                            "upscaled to guide_size before encoding for higher quality, then "
                            "composited back at original resolution. Takes precedence over latent."
                        ),
                    },
                ),
                "latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Pre-encoded latent to use directly, skipping the VAE encode step. "
                            "Useful when chaining from WanVaceToVideo or another latent-output node. "
                            "Ignored if image_frames is also connected."
                        ),
                    },
                ),
                "mask_opt": ("MASK",),
                "guide_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Target minimum side for encoding when image_frames is used. Ignored for latent input.",
                    },
                ),
                "max_size": (
                    "INT",
                    {
                        "default": 1536,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Caps the encoding resolution when image_frames is used. Ignored for latent input.",
                    },
                ),
                "upscale_model": (
                    "UPSCALE_MODEL",
                    {
                        "tooltip": (
                            "Optional upscale model (e.g. ESRGAN) applied before encoding when "
                            "image_frames is used. Frames are scaled back to original resolution after decoding."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "LATENT")
    RETURN_NAMES = ("image", "mask", "latent")
    FUNCTION = "execute"
    CATEGORY = "link/video"

    DESCRIPTION = (
        "Temporal gap filler for WAN T2V models. Encodes all frames together, "
        "adds noise only to masked frames, and lets the model's temporal attention "
        "predict what the masked frames should look like given the surrounding context. "
        "Accepts either raw image_frames (with optional upscaling) or a pre-encoded "
        "latent directly. Any conditioning set up upstream passes through unchanged."
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
                    f"mask_opt has {mask.shape[0]} frames but video has {frame_count}."
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
    def _resize_phantom_in_conditioning(
        conditioning: list, latent_height: int, latent_width: int, drop: bool = False
    ) -> list:
        """Resize time_dim_concat spatial dims to match the latent being sampled.

        Pass drop=True for models whose patch_embedding in_dim > 16: they use
        channel-wise phantom concatenation via concat_cond, making time_dim_concat
        (which is always 16-channel) incompatible — the temporal cat would try to
        join a 36-channel x with a 16-channel phantom and crash.
        """
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
    def _resize_vace_in_conditioning(
        conditioning: list,
        latent_height: int,
        latent_width: int,
    ) -> list:
        """Resize vace_frames and vace_mask latents to match a new spatial resolution."""
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
    def _upscale_frames_with_model(
        upscale_model,
        images: torch.Tensor,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        upscaled = ImageUpscaleWithModel().upscale(upscale_model, images)[0]
        upscaled = upscaled.to(device=images.device, dtype=torch.float32)
        if upscaled.shape[1] != target_height or upscaled.shape[2] != target_width:
            upscaled = VideoDetailer._resize_images(
                upscaled, target_height, target_width
            )
        return upscaled

    @staticmethod
    def _build_noise_mask(
        mask: torch.Tensor,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        """Downsample pixel-space frame mask to latent space.

        Shape returned: [1, 1, latent_frames, latent_height, latent_width]
        Max-pools temporally so any masked pixel frame marks the whole latent frame.
        """
        latent_mask = F.interpolate(
            mask.unsqueeze(1),
            size=(latent_height, latent_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return F.adaptive_max_pool3d(
            latent_mask.unsqueeze(0).unsqueeze(0),
            output_size=(latent_frames, latent_height, latent_width),
        )

    @staticmethod
    def _build_vace_conditioning(
        vae,
        frames: torch.Tensor,  # [T, H, W, 3] at encoding resolution, float32 [0,1]
        mask: torch.Tensor,    # [T, H, W] at encoding resolution, float32 [0,1]
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build vace_frames (32ch) and vace_mask (64ch) for VACE conditioning.

        Mirrors the inactive/reactive encoding in WanVaceToVideo so the VACE
        feature-injection blocks receive accurate, frame-aligned reference:
          - inactive channels: original content where mask=0 (reference frames), gray where mask=1
          - reactive channels: original content where mask=1 (to-generate frames), gray where mask=0
          - vace_mask: 64-channel stride-packed mask matching WAN's latent packing convention

        Returns:
            vace_frames: [1, 32, T_lat, H_lat, W_lat] on device
            vace_mask:   [1, 64, T_lat, H_lat, W_lat] on device
        """
        T, H, W = mask.shape
        vae_stride = 8
        H_lat, W_lat = H // vae_stride, W // vae_stride

        mask_3ch = mask.unsqueeze(-1).expand(-1, -1, -1, 3)

        frames_c = frames - 0.5
        inactive = (frames_c * (1.0 - mask_3ch) + 0.5).clamp(0.0, 1.0)
        reactive = (frames_c * mask_3ch + 0.5).clamp(0.0, 1.0)

        inactive_lat = vae.encode(inactive[:, :, :, :3]).to(device)
        reactive_lat = vae.encode(reactive[:, :, :, :3]).to(device)
        vace_frames = torch.cat([inactive_lat, reactive_lat], dim=1)

        T_lat = vace_frames.shape[2]

        # Stride-pack spatial dims into 64 channels then resize temporally,
        # matching WanVaceToVideo's mask encoding exactly.
        m = mask.view(T, H_lat, vae_stride, W_lat, vae_stride)
        m = m.permute(2, 4, 0, 1, 3).reshape(vae_stride * vae_stride, T, H_lat, W_lat)
        m = F.interpolate(
            m.unsqueeze(0).float(),
            size=(T_lat, H_lat, W_lat),
            mode="nearest-exact",
        ).squeeze(0)

        return vace_frames, m.unsqueeze(0).to(device)

    @staticmethod
    def _cleanup():
        comfy.model_management.soft_empty_cache()
        gc.collect()

    def execute(
        self,
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
        image_frames=None,
        latent=None,
        mask_opt=None,
        guide_size=1024,
        max_size=1536,
        upscale_model=None,
        **_kwargs,
    ):
        if image_frames is None and latent is None:
            raise ValueError("Connect either image_frames or latent — at least one is required.")

        device = comfy.model_management.get_torch_device()

        # --- Build starting latent and pixel frames ---
        if image_frames is not None:
            # Image path: encode with optional upscaling for quality.
            frames = image_frames.to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
            frame_count, height, width, _ = frames.shape

            if frame_count < 5:
                raise ValueError(
                    f"image_frames has {frame_count} frame(s); minimum is 5."
                )
            if isinstance(model.model, comfy.model_base.WAN21) and (frame_count - 1) % 4 != 0:
                rem = (frame_count - 1) % 4
                lower = frame_count - rem
                upper = lower + 4
                raise ValueError(
                    f"{frame_count} frames is not valid for WAN (must satisfy 1 + n×4). "
                    f"Nearest valid counts: {lower} or {upper}."
                )
            mask = self._prepare_mask(mask_opt, frame_count, height, width, device)

            target_width, target_height = self._pick_target_size(height, width, guide_size, max_size)
            if upscale_model is not None:
                frames_target = self._upscale_frames_with_model(upscale_model, frames, target_height, target_width)
            else:
                frames_target = self._resize_images(frames, target_height, target_width)

            mask_target = self._resize_masks(mask, target_height, target_width).clamp_(0.0, 1.0)

            mask_for_noise = self._gaussian_blur_mask(
                mask_target, noise_mask_feather
            ).clamp_(0.0, 1.0)

            # Gray out masked-frame regions before encoding.  WAN's temporal VAE uses
            # causal convolutions, so bad original content in frame N bleeds into frame
            # N+1's latent, N+1 into N+2, etc.  This directional bleed-through is the
            # exact cause of the quality ramp-up across masked frames: the first masked
            # frame carries the most contamination, the last the least.  Replacing masked
            # regions with neutral gray (0.5) breaks the contamination chain and gives
            # all masked frames a clean, equivalent starting point in latent space.
            frames_for_latent = (
                frames_target * (1.0 - mask_target.unsqueeze(-1))
                + 0.5 * mask_target.unsqueeze(-1)
            )
            all_latent = vae.encode(frames_for_latent[:, :, :, :3]).to(device)

        else:
            # Latent path: use provided latent directly, decode once for pixel reference.
            all_latent = latent["samples"].to(device=device, dtype=torch.float32)

            decoded = vae.decode(all_latent).to(device=device, dtype=torch.float32)
            if decoded.ndim == 5:
                decoded = decoded.squeeze(0)
            frames = decoded.clamp_(0.0, 1.0)
            frame_count, height, width, _ = frames.shape

            if frame_count < 5:
                raise ValueError(
                    f"Decoded latent has {frame_count} frame(s); minimum is 5."
                )
            if isinstance(model.model, comfy.model_base.WAN21) and (frame_count - 1) % 4 != 0:
                rem = (frame_count - 1) % 4
                lower = frame_count - rem
                upper = lower + 4
                raise ValueError(
                    f"{frame_count} decoded frames is not valid for WAN (must satisfy 1 + n×4). "
                    f"Nearest valid counts: {lower} or {upper}."
                )
            mask = self._prepare_mask(mask_opt, frame_count, height, width, device)
            mask_for_noise = self._gaussian_blur_mask(mask, noise_mask_feather).clamp_(0.0, 1.0)

        latent_height = all_latent.shape[-2]
        latent_width = all_latent.shape[-1]

        # --- Prepare conditioning ---
        positive_cond = self._slice_conditioning_for_chunk(positive, 0, frame_count)
        negative_cond = self._slice_conditioning_for_chunk(negative, 0, frame_count)

        # Channel-wise phantom models (in_dim > 16, e.g. Phantom Pure Fusionix) expand x to
        # 36 channels via concat_cond before _forward runs.  time_dim_concat is always 16-channel
        # so the temporal cat at model.py:657 would mismatch.  Drop it and let concat_cond handle
        # phantom conditioning channel-wise instead.
        patch_emb = getattr(getattr(model.model, 'diffusion_model', None), 'patch_embedding', None)
        drop_time_dim = (
            patch_emb is not None
            and hasattr(patch_emb, 'weight')
            and patch_emb.weight.shape[1] > 16
        )
        positive_cond = self._resize_phantom_in_conditioning(positive_cond, latent_height, latent_width, drop=drop_time_dim)
        negative_cond = self._resize_phantom_in_conditioning(negative_cond, latent_height, latent_width, drop=drop_time_dim)
        positive_cond = self._resize_vace_in_conditioning(positive_cond, latent_height, latent_width)
        negative_cond = self._resize_vace_in_conditioning(negative_cond, latent_height, latent_width)

        start_step = int(steps * (1.0 - denoise))

        # --- Sample ---
        if image_frames is not None:
            composite_mask = self._gaussian_blur_mask(mask, feather).clamp_(0.0, 1.0)
            if composite_mask.amax() < 1e-6:
                return (frames.cpu(), mask.cpu(), {"samples": all_latent.cpu()})

            # Rebuild VACE conditioning from the actual encoded frames so the VACE
            # feature-injection blocks get frame-accurate reference.  Surrounding good
            # frames get mask=0 (VACE reference) and masked frames get mask=1 (generate).
            # This replaces any upstream vace_frames from WanVaceToVideo, which may have
            # been encoded at a different resolution or from a different pass.
            if isinstance(model.model, comfy.model_base.WAN21_Vace):
                import node_helpers
                vace_f, vace_m = self._build_vace_conditioning(
                    vae, frames_target, mask_target, device
                )
                _vace_vals = {"vace_frames": [vace_f], "vace_mask": [vace_m], "vace_strength": [1.0]}
                positive_cond = node_helpers.conditioning_set_values(positive_cond, _vace_vals)
                negative_cond = node_helpers.conditioning_set_values(negative_cond, _vace_vals)

            if "denoise_mask_function" not in model.model_options:
                model = DifferentialDiffusion.execute(model)[0]

            noise_mask = self._build_noise_mask(
                mask_for_noise, all_latent.shape[2], latent_height, latent_width
            )
            latent_in = {"samples": all_latent, "noise_mask": noise_mask}
        else:
            # Latent path: the input latent from WanVaceToVideo is zeros (noise
            # initialization), NOT encoded video content.  Decoding zeros gives
            # garbage pixels, so we cannot composite against it.  DifferentialDiffusion
            # with a zero latent would "preserve" those zeros as unmasked frame output,
            # producing garbage for good frames.  Skip both: regenerate all frames
            # from the conditioning, then return the decoded result directly.
            composite_mask = None
            latent_in = {"samples": all_latent}

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
            latent_in,
            start_step,
            steps,
            "disable",
        )[0]

        sampled_latent = sampled["samples"]

        # --- Decode ---
        refined = vae.decode(sampled_latent).to(device=device, dtype=torch.float32)
        if refined.ndim == 5:
            refined = refined.squeeze(0)
        refined = refined.clamp_(0.0, 1.0)

        # --- Composite (image path only) ---
        if composite_mask is not None:
            if refined.shape[1] != height or refined.shape[2] != width:
                refined = self._resize_images(refined, height, width, mode="bicubic")

            # WAN's temporal VAE may decode fewer frames than were encoded.
            n_refined = refined.shape[0]
            if n_refined < frame_count:
                refined = torch.cat([refined, frames[n_refined:]], dim=0)
            elif n_refined > frame_count:
                refined = refined[:frame_count]

            composite_mask_4d = composite_mask.unsqueeze(-1)
            output = (1.0 - composite_mask_4d) * frames + composite_mask_4d * refined
        else:
            output = refined

        self._cleanup()
        return (output.clamp(0.0, 1.0).cpu(), mask.cpu(), {"samples": sampled_latent.cpu()})
