"""Tiled video detailer — processes video frames tile by tile with an optional
per-tile reference image, then blends tiles back using feathered seam weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.samplers
import nodes
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion


class VideoTileDetailer:
    """Tile-based video detailer.

    Decodes all frames, then for each tile region:
    - optionally encodes the matching crop of a reference image and prepends it
      as a protected temporal frame so the model has spatial context
    - strips VACE/phantom conditioning (vace_frames, time_dim_concat) so the
      patched WanVaceAdvanced forward is not triggered and frame counts stay valid
    - denoises with partial noise controlled by denoise
    - decodes and accumulates with cosine feather weights for seamless blending
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "basic_pipe": ("BASIC_PIPE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How much to re-denoise each tile. 0 = no change, 1 = full re-generation.",
                    },
                ),
                "tile_width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Tile width in pixels. Must be a multiple of 8.",
                    },
                ),
                "tile_height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Tile height in pixels. Must be a multiple of 8.",
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 512,
                        "step": 8,
                        "tooltip": "Overlap between adjacent tiles in pixels.",
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Width of the cosine blend ramp at tile edges for seamless stitching.",
                    },
                ),
            },
            "optional": {
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": "Cropped to each tile region and prepended as a "
                        "protected temporal context frame. Guides tile refinement "
                        "toward the reference without changing frame count."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Tile-based video detailer. Optional reference image is cropped to each tile "
        "region and used as a protected temporal context frame. VACE/phantom "
        "conditioning is stripped per-tile to avoid frame-count mismatches. "
        "Blends tiles back with cosine feathering."
    )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _fix_decoded(decoded: torch.Tensor, expected_h: int) -> torch.Tensor:
        """Normalise VAE output to (N, H, W, C)."""
        if decoded.ndim == 5:
            b, f, d1, d2, c = decoded.shape
            decoded = decoded.reshape(b * f, d1, d2, c)
        if decoded.shape[1] != expected_h:
            decoded = torch.rot90(decoded, k=3, dims=(1, 2)).contiguous()
        return decoded

    @staticmethod
    def _get_tiles(
        H: int, W: int, tile_h: int, tile_w: int, overlap: int
    ) -> list[tuple[int, int, int, int]]:
        """Return (y1, x1, y2, x2) tile regions covering [0,H)×[0,W)."""
        if H <= tile_h and W <= tile_w:
            return [(0, 0, H, W)]

        stride_h = max(1, tile_h - overlap)
        stride_w = max(1, tile_w - overlap)

        def positions(dim, tile, stride):
            pts = list(range(0, max(1, dim - tile), stride))
            last = max(0, dim - tile)
            if not pts or pts[-1] != last:
                pts.append(last)
            return sorted(set(pts))

        ys = positions(H, tile_h, stride_h)
        xs = positions(W, tile_w, stride_w)
        return [
            (y, x, min(y + tile_h, H), min(x + tile_w, W))
            for y in ys
            for x in xs
        ]

    @staticmethod
    def _feather_weight(
        tile_h: int, tile_w: int, feather: int, device: torch.device
    ) -> torch.Tensor:
        """Cosine ramp weight (tile_h, tile_w) — full in centre, fades at edges."""
        w = torch.ones(tile_h, tile_w, device=device)
        f = min(feather, tile_h // 2, tile_w // 2)
        if f > 0:
            ramp = (1 - torch.cos(torch.linspace(0, torch.pi, f + 2, device=device)[1:-1])) / 2
            w[:f, :] *= ramp[:, None]
            w[-f:, :] *= ramp.flip(0)[:, None]
            w[:, :f] *= ramp[None, :]
            w[:, -f:] *= ramp.flip(0)[None, :]
        return w

    @staticmethod
    def _strip_tile_conditioning(cond_list: list) -> list:
        """Strip VACE and phantom conditioning for per-tile sampling.

        Spatially-cropped vace_frames cause ghosting (different reference
        characters per tile). time_dim_concat phantom frames cause temporal
        count mismatches. Stripping both lets the model refine tiles using
        only text conditioning + the initial tile latent structure.
        """
        stripped = []
        for (t, d) in cond_list:
            d = d.copy()
            for key in ("vace_frames", "vace_mask", "vace_strength", "time_dim_concat"):
                d.pop(key, None)
            stripped.append((t, d))
        return stripped

    # ------------------------------------------------------------------ main

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
        tile_width,
        tile_height,
        tile_overlap,
        feather,
        reference_image=None,
    ):
        model, _clip, vae, positive, negative = basic_pipe
        device = comfy.model_management.get_torch_device()

        lat = latent["samples"]  # (1, C, T, H_lat, W_lat)
        H_lat, W_lat = lat.shape[3], lat.shape[4]
        img_h, img_w = H_lat * 8, W_lat * 8

        # Decode all frames to pixel space
        decoded = vae.decode(lat)
        frames = self._fix_decoded(decoded, img_h).to(device)  # (N_total, H, W, C)
        N_total = frames.shape[0]

        # Phantom frames (WanVacePhantomSimpleV2) are stored in the latent, so
        # vae.decode gives more frames than the actual video. Strip them by
        # inferring phantom latent frames from time_dim_concat in conditioning.
        T_total = lat.shape[2]
        T_phantom_lat = 0
        for (_t, d) in positive:
            tdc = d.get("time_dim_concat")
            if isinstance(tdc, torch.Tensor):
                T_phantom_lat = tdc.shape[2]
                break

        T_video_lat = T_total - T_phantom_lat
        if T_phantom_lat > 0 and T_total > 1 and N_total > 1:
            temporal_ratio = round((N_total - 1) / (T_total - 1))
            N_video = (T_video_lat - 1) * temporal_ratio + 1
            frames = frames[:N_video]
            print(
                f"[VideoTileDetailer] phantom: {T_phantom_lat} latent frames stripped, "
                f"keeping {N_video}/{N_total} pixel frames"
            )
        N = frames.shape[0]
        print(f"[VideoTileDetailer] {N} frames {img_w}×{img_h}")

        # Prepare full-res reference image (cropped per-tile later)
        ref = None
        if reference_image is not None:
            ref = (reference_image[0] if reference_image.ndim == 4 else reference_image).to(device)
            if ref.shape[0] != img_h or ref.shape[1] != img_w:
                ref = (
                    F.interpolate(
                        ref.unsqueeze(0).permute(0, 3, 1, 2),
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                )
            # Apply DifferentialDiffusion once so noise_mask protects the ref frame
            if "denoise_mask_function" not in model.model_options:
                model = DifferentialDiffusion.execute(model)[0]
            print(f"[VideoTileDetailer] reference image: {ref.shape}")

        tiles = self._get_tiles(img_h, img_w, tile_height, tile_width, tile_overlap)
        print(
            f"[VideoTileDetailer] {len(tiles)} tiles "
            f"({tile_width}×{tile_height}, overlap={tile_overlap})"
        )

        # start_step derived from denoise: denoise=1.0 → start_step=0, denoise=0 → start_step=steps
        start_step = int(steps * (1.0 - denoise))

        # Accumulation buffers in pixel space
        out_acc = torch.zeros(N, img_h, img_w, frames.shape[-1], device=device)
        out_wgt = torch.zeros(N, img_h, img_w, 1, device=device)

        for idx, (y1, x1, y2, x2) in enumerate(tiles):
            th, tw = y2 - y1, x2 - x1
            print(f"[VideoTileDetailer] tile {idx+1}/{len(tiles)} ({x1},{y1})→({x2},{y2})")

            # Crop tile frames (all N video frames, tile spatial region)
            tile_frames = frames[:, y1:y2, x1:x2, :].contiguous()  # (N, th, tw, C)

            # Encode video tile frames
            tile_lat = vae.encode(tile_frames).to(device)  # (1, C, T, h_lat, w_lat)
            vid_T = tile_lat.shape[2]
            h_lat, w_lat = tile_lat.shape[3], tile_lat.shape[4]

            if ref is not None:
                # Encode matching crop of the reference image as a temporal context frame
                ref_tile = ref[y1:y2, x1:x2, :].contiguous()  # (th, tw, C)
                ref_lat = vae.encode(ref_tile.unsqueeze(0)).to(device)  # (1, C, ref_T, h, w)
                ref_T = ref_lat.shape[2]

                # Prepend reference temporally; noise_mask protects it (=0)
                combined = torch.cat([ref_lat, tile_lat], dim=2)
                ref_mask = torch.zeros(1, 1, ref_T, h_lat, w_lat, device=device)
                vid_mask = torch.ones(1, 1, vid_T, h_lat, w_lat, device=device)
                noise_mask = torch.cat([ref_mask, vid_mask], dim=2)
                latent_in = {"samples": combined, "noise_mask": noise_mask}
            else:
                ref_T = 0
                latent_in = {"samples": tile_lat}

            # Strip VACE/phantom conditioning. With time_dim_concat removed the
            # WanVaceAdvanced patched forward won't trigger, so prepending ref_T
            # extra temporal frames is safe (no reference_frames < 0 check).
            pos_tile = self._strip_tile_conditioning(positive)
            neg_tile = self._strip_tile_conditioning(negative)

            sampled = nodes.NODE_CLASS_MAPPINGS["KSamplerAdvanced"]().sample(
                model,
                "enable",     # add_noise
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                pos_tile,
                neg_tile,
                latent_in,
                start_step,
                steps,        # end_step
                "disable",    # return_with_leftover_noise
            )[0]

            # Strip reference temporal frames from sampler output, then decode
            video_lat = sampled["samples"][:, :, ref_T:, :, :]
            tile_decoded = vae.decode(video_lat)
            refined = self._fix_decoded(tile_decoded, th).to(device)  # (N, th, tw, C)

            out_n = min(N, refined.shape[0])

            # Feathered blend weight
            weight = self._feather_weight(th, tw, feather, device)  # (th, tw)
            w4d = weight.unsqueeze(0).unsqueeze(-1)                  # (1, th, tw, 1)

            out_acc[:out_n, y1:y2, x1:x2, :] += refined[:out_n] * w4d
            out_wgt[:out_n, y1:y2, x1:x2, :] += w4d

        result = (out_acc / out_wgt.clamp(min=1e-8)).clamp(0, 1).cpu()
        print(f"[VideoTileDetailer] done → {result.shape}")
        return (result,)
