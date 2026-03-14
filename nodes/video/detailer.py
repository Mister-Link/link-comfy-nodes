from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.samplers
import node_helpers
import nodes
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion


class VideoDetailer:
    _MASK_EPSILON = 1e-6

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
                        "tooltip": "Gaussian blur for the inpainting mask boundary.",
                    },
                ),
            },
            "optional": {
                "mask_opt": ("MASK",),
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image prepended as a temporal context "
                        "frame. Protected from denoising — the model attends to "
                        "it while generating all video frames at native resolution."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Reference-guided video detailer. Prepends reference as a temporal "
        "context frame at native resolution, denoises video frames, then "
        "composites back."
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
        ref_frame,  # (H, W, 3)
        video_frames,  # (N, H, W, 3)
        mask,  # (N, H, W)  [0=keep, 1=inpaint]
        vae,
        latent_frames,  # T — video latent temporal frames
        ref_lat_T,  # reference latent temporal frames (usually 1)
        img_height,
        img_width,
        device,
    ):
        vae_stride = 8
        H_lat = img_height // vae_stride
        W_lat = img_width // vae_stride
        N = video_frames.shape[0]

        # ---- Reference frame: inactive=ref, reactive=neutral(0.5), mask=0 ----
        ref_exp = ref_frame.unsqueeze(0)  # (1, H, W, 3)
        ref_neutral = torch.full_like(ref_exp, 0.5)
        ref_inactive_lat = vae.encode(ref_exp[:, :, :, :3]).to(device)
        ref_reactive_lat = vae.encode(ref_neutral[:, :, :, :3]).to(device)
        ref_vace_lat = torch.cat([ref_inactive_lat, ref_reactive_lat], dim=1)
        ref_vace_mask = torch.zeros(1, 64, ref_lat_T, H_lat, W_lat, device=device)

        # ---- Video frames: standard VACE inpainting ----
        mask_4d = mask.unsqueeze(-1)  # (N, H, W, 1)
        greyed = video_frames * (1 - mask_4d) + 0.5 * mask_4d
        control = greyed - 0.5
        inactive = (control * (1 - mask_4d)) + 0.5
        reactive = (control * mask_4d) + 0.5

        inactive_lat = vae.encode(inactive[:, :, :, :3]).to(device)
        reactive_lat = vae.encode(reactive[:, :, :, :3]).to(device)
        video_vace_lat = torch.cat([inactive_lat, reactive_lat], dim=1)

        # VACE mask: 64 channels from vae_stride² pixel sub-blocks
        mask_blocks = mask.view(N, H_lat, vae_stride, W_lat, vae_stride)
        mask_blocks = mask_blocks.permute(2, 4, 0, 1, 3)
        mask_blocks = mask_blocks.reshape(vae_stride * vae_stride, N, H_lat, W_lat)
        mask_blocks = F.interpolate(
            mask_blocks.unsqueeze(0),
            size=(latent_frames, H_lat, W_lat),
            mode="nearest-exact",
        ).squeeze(0)
        video_vace_mask = mask_blocks.unsqueeze(0)  # (1, 64, T, H_lat, W_lat)

        # Combine ref + video temporally
        control_latent = torch.cat([ref_vace_lat, video_vace_lat], dim=2)
        vace_mask = torch.cat([ref_vace_mask, video_vace_mask], dim=2)

        print(
            f"[Video Detailer] VACE latent {control_latent.shape} "
            f"mask {vace_mask.shape} mean={vace_mask.mean():.3f}"
        )
        return [control_latent], [vace_mask]

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
        model, _clip, vae, positive, negative = basic_pipe

        device = comfy.model_management.get_torch_device()

        latent_samples = latent["samples"]  # (1, C, T, H_lat, W_lat)
        latent_frames = latent_samples.shape[2]
        H_lat = latent_samples.shape[3]
        W_lat = latent_samples.shape[4]
        img_height = H_lat * 8
        img_width = W_lat * 8

        print(
            f"[Video Detailer] latent {latent_samples.shape} "
            f"-> {img_width}x{img_height} px, device={device}"
        )

        # Decode video frames (needed for compositing and VACE conditioning)
        original_decoded = vae.decode(latent_samples)
        original_frames = self._fix_decoded_shape(original_decoded, img_height).to(
            device
        )
        num_pixel_frames = original_frames.shape[0]
        print(
            f"[Video Detailer] decoded {num_pixel_frames} frames {original_frames.shape}"
        )

        # Build inpainting mask
        if mask_opt is None:
            mask = torch.ones(
                (num_pixel_frames, img_height, img_width),
                dtype=torch.float32,
                device=device,
            )
        else:
            mask = mask_opt.clone().to(device=device, dtype=torch.float32)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).expand(num_pixel_frames, -1, -1).contiguous()
            elif mask.ndim != 3:
                raise ValueError(
                    f"mask_opt must be 2D or 3D, got shape {tuple(mask.shape)}"
                )
            elif mask.shape[0] == 1:
                mask = mask[0:1].expand(num_pixel_frames, -1, -1).contiguous()
            elif mask.shape[0] != num_pixel_frames:
                raise ValueError(
                    "mask_opt frame count must be 1 or match the decoded frame count; "
                    f"got mask batch {mask.shape[0]} for {num_pixel_frames} frames"
                )

        mask = mask.clamp_(0.0, 1.0)
        mask_frame_max = mask.flatten(1).amax(dim=1)
        passthrough_frames = mask_frame_max <= self._MASK_EPSILON
        if passthrough_frames.any():
            mask[passthrough_frames] = 0.0
            print(
                "[Video Detailer] passthrough frames="
                f"{int(passthrough_frames.sum().item())}/{mask.shape[0]}"
            )

        # Get reference frame
        if reference_image is not None:
            ref = (
                reference_image[0] if reference_image.ndim == 4 else reference_image
            ).to(device)
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

        # Encode reference as a single temporal latent
        ref_latent = vae.encode(ref.unsqueeze(0)).to(
            device
        )  # (1, C, ref_T, H_lat, W_lat)
        ref_lat_T = ref_latent.shape[2]
        print(f"[Video Detailer] ref_latent {ref_latent.shape}")

        # Combined latent: [ref_temporal | video_temporal]
        combined_latent = torch.cat([ref_latent, latent_samples.to(device)], dim=2)
        print(f"[Video Detailer] combined_latent {combined_latent.shape}")

        # Build noise_mask in latent space: (1, 1, ref_lat_T + T, H_lat, W_lat)
        # Reference positions → 0 (protected), video positions → inpainting mask
        mask_blurred = self._gaussian_blur_mask(mask, noise_mask_feather)  # (N, H, W)
        mask_lat = F.interpolate(
            mask_blurred.unsqueeze(1),
            size=(H_lat, W_lat),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (N, H_lat, W_lat)
        # Resample temporal axis: N pixel frames → latent_frames latent frames
        mask_lat_t = F.interpolate(
            mask_lat.unsqueeze(0).unsqueeze(0),
            size=(latent_frames, H_lat, W_lat),
            mode="nearest-exact",
        )  # (1, 1, T, H_lat, W_lat)
        ref_noise_mask = torch.zeros(1, 1, ref_lat_T, H_lat, W_lat, device=device)
        noise_mask_5d = torch.cat([ref_noise_mask, mask_lat_t], dim=2)
        print(
            f"[Video Detailer] noise_mask {noise_mask_5d.shape} "
            f"mean={noise_mask_5d.mean():.3f}"
        )

        # VACE conditioning (if present in the conditioning)
        has_vace = (
            len(positive) > 0
            and len(positive[0]) > 1
            and positive[0][1].get("vace_strength", None) is not None
        )
        if has_vace:
            vace_strength = positive[0][1]["vace_strength"][0]
            print(
                f"[Video Detailer] VACE strength={vace_strength}, building temporal context"
            )
            vace_frames_list, vace_mask_list = self._build_vace_conditioning(
                ref,
                original_frames,
                mask,
                vae,
                latent_frames,
                ref_lat_T,
                img_height,
                img_width,
                device,
            )
            vace_values = {
                "vace_frames": vace_frames_list,
                "vace_mask": vace_mask_list,
                "vace_strength": [vace_strength],
            }
            positive = node_helpers.conditioning_set_values(positive, vace_values)
            negative = node_helpers.conditioning_set_values(negative, vace_values)
            print("[Video Detailer] VACE temporal conditioning applied")
        else:
            print("[Video Detailer] no VACE detected, using temporal ref only")

        if "denoise_mask_function" not in model.model_options:
            model = DifferentialDiffusion.execute(model)[0]
            print("[Video Detailer] DifferentialDiffusion applied")

        latent_dict = {
            "samples": combined_latent,
            "noise_mask": noise_mask_5d,
        }

        samples = nodes.NODE_CLASS_MAPPINGS["KSamplerAdvanced"]().sample(
            model,
            "enable",
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_dict,
            start_step,
            end_step,
            "disable",
        )[0]

        # Extract only the video portion of the output (skip reference temporal frames)
        video_output_latent = samples["samples"][:, :, ref_lat_T:, :, :]
        decoded_video = vae.decode(video_output_latent)
        refined_frames = self._fix_decoded_shape(decoded_video, img_height).to(device)
        print(f"[Video Detailer] refined {refined_frames.shape}")

        # Composite refined frames back onto originals using the inpainting mask
        composite_mask = self._gaussian_blur_mask(mask, feather)
        mask_4d = composite_mask.unsqueeze(-1)

        out_n = min(original_frames.shape[0], refined_frames.shape[0])
        if original_frames.shape[0] != refined_frames.shape[0]:
            print(
                f"[Video Detailer] WARNING: frame count mismatch — original={original_frames.shape[0]} refined={refined_frames.shape[0]}, compositing {out_n} frames"
            )
        output = (1 - mask_4d[:out_n]) * original_frames[:out_n] + mask_4d[
            :out_n
        ] * refined_frames[:out_n]
        if passthrough_frames[:out_n].any():
            output[passthrough_frames[:out_n]] = original_frames[:out_n][
                passthrough_frames[:out_n]
            ]
        print(f"[Video Detailer] output {output.shape}")

        output_mask = mask
        return (output, output_mask)
