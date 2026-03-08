"""Tiled video detailer — processes video frames tile by tile with reference-guided
consistency, then blends tiles back using feathered seam weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.samplers
import nodes
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion


class VideoTileDetailer:
    """Tile-based video detailer.

    Decodes all frames, then for each tile region across the whole video clip:
    - crops the reference image to that region for spatially-aware context
    - encodes tile frames + reference tile together
    - denoises with the reference temporal frame protected (noise_mask=0)
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
                        "tooltip": "Tile width in pixels. Should be a multiple of 8.",
                    },
                ),
                "tile_height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Tile height in pixels. Should be a multiple of 8.",
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
                        "tooltip": "Reference frame cropped to each tile region for "
                        "spatially-aware consistency. Falls back to first video frame."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Tile-based video detailer. Processes each spatial tile across all video frames "
        "with a reference image for consistency, then blends tiles back seamlessly."
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
            ramp = (1 - torch.cos(torch.linspace(0, torch.pi, f + 2)[1:-1])) / 2
            w[:f, :] *= ramp[:, None]
            w[-f:, :] *= ramp.flip(0)[:, None]
            w[:, :f] *= ramp[None, :]
            w[:, -f:] *= ramp.flip(0)[None, :]
        return w

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
        frames = self._fix_decoded(decoded, img_h).to(device)  # (N, H, W, C)
        N = frames.shape[0]
        print(f"[VideoTileDetailer] {N} frames {img_w}×{img_h}")

        # Prepare full-res reference (cropped per-tile later)
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
        else:
            ref = frames[0]  # (H, W, C)

        # DifferentialDiffusion so noise_mask drives per-position denoise depth
        if "denoise_mask_function" not in model.model_options:
            model = DifferentialDiffusion.execute(model)[0]

        tiles = self._get_tiles(img_h, img_w, tile_height, tile_width, tile_overlap)
        print(
            f"[VideoTileDetailer] {len(tiles)} tiles "
            f"({tile_width}×{tile_height}, overlap={tile_overlap})"
        )

        # start_step derived from denoise: 0 denoise → start at end, 1 → start at 0
        start_step = int(steps * (1.0 - denoise))

        # Accumulation buffers in pixel space
        out_acc = torch.zeros(N, img_h, img_w, frames.shape[-1], device=device)
        out_wgt = torch.zeros(N, img_h, img_w, 1, device=device)

        for idx, (y1, x1, y2, x2) in enumerate(tiles):
            th, tw = y2 - y1, x2 - x1
            print(f"[VideoTileDetailer] tile {idx+1}/{len(tiles)} ({x1},{y1})→({x2},{y2})")

            # Crop tile from all video frames and the reference
            tile_frames = frames[:, y1:y2, x1:x2, :].contiguous()  # (N, th, tw, C)
            ref_tile = ref[y1:y2, x1:x2, :].contiguous()           # (th, tw, C)

            # Encode reference tile (single temporal frame)
            ref_lat = vae.encode(ref_tile.unsqueeze(0)).to(device)  # (1, C, ref_T, h, w)
            ref_T = ref_lat.shape[2]

            # Encode all video frames for this tile
            tile_lat = vae.encode(tile_frames).to(device)  # (1, C, vid_T, h, w)
            vid_T = tile_lat.shape[2]
            h_lat = tile_lat.shape[3]
            w_lat = tile_lat.shape[4]

            # Concatenate ref + video along temporal axis
            combined = torch.cat([ref_lat, tile_lat], dim=2)  # (1, C, ref_T+vid_T, h, w)

            # Noise mask: 0 = protected (reference), 1 = denoise (video)
            ref_mask = torch.zeros(1, 1, ref_T, h_lat, w_lat, device=device)
            vid_mask = torch.ones(1, 1, vid_T, h_lat, w_lat, device=device)
            noise_mask = torch.cat([ref_mask, vid_mask], dim=2)

            latent_in = {"samples": combined, "noise_mask": noise_mask}

            sampled = nodes.NODE_CLASS_MAPPINGS["KSamplerAdvanced"]().sample(
                model,
                "enable",   # add_noise
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_in,
                start_step,
                steps,      # end_step
                "disable",  # return_with_leftover_noise
            )[0]

            # Strip reference temporal frames from sampler output
            video_lat = sampled["samples"][:, :, ref_T:, :, :]

            # Decode tile
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
