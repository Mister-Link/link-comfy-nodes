"""Tiled video detailer — processes video frames tile by tile with VACE conditioning
spatially cropped per tile, then blends tiles back using feathered seam weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.samplers
import nodes


class VideoTileDetailer:
    """Tile-based video detailer.

    Decodes all frames, then for each tile region across the whole video clip:
    - encodes tile frames (all T frames, same temporal count as original latent)
    - crops VACE conditioning tensors spatially to the tile region so temporal
      frame counts stay consistent (avoids 'reference_frames cannot be negative')
    - denoises with partial noise controlled by the denoise parameter
    - decodes and accumulates with cosine feather weights for seamless blending

    Works with WanVacePhantomSimpleV2 and other VACE conditioning setups.
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
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Tile-based video detailer. Spatially crops VACE conditioning per tile so "
        "temporal frame counts stay consistent with WanVace/Phantom workflows. "
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
    def _crop_conditioning(
        cond_list: list,
        lat_y1: int,
        lat_x1: int,
        lat_y2: int,
        lat_x2: int,
    ) -> list:
        """Crop VACE-related conditioning tensors to the tile's latent spatial region.

        Crops the last two dimensions (H_lat, W_lat) of:
        - vace_frames  (list of tensors: ..., H, W)
        - vace_mask    (list of tensors: ..., H, W)
        - time_dim_concat  (tensor: ..., H, W)
        """
        cropped = []
        for (t, d) in cond_list:
            d = d.copy()

            if "vace_frames" in d:
                d["vace_frames"] = [
                    f[..., lat_y1:lat_y2, lat_x1:lat_x2] for f in d["vace_frames"]
                ]

            if "vace_mask" in d:
                d["vace_mask"] = [
                    m[..., lat_y1:lat_y2, lat_x1:lat_x2] for m in d["vace_mask"]
                ]

            if "time_dim_concat" in d and isinstance(d["time_dim_concat"], torch.Tensor):
                d["time_dim_concat"] = d["time_dim_concat"][..., lat_y1:lat_y2, lat_x1:lat_x2]

            cropped.append((t, d))
        return cropped

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

            # Latent tile coordinates (VAE spatial stride = 8)
            lat_y1, lat_x1 = y1 // 8, x1 // 8
            lat_y2, lat_x2 = y2 // 8, x2 // 8

            # Crop tile frames (all N video frames, tile spatial region)
            tile_frames = frames[:, y1:y2, x1:x2, :].contiguous()  # (N, th, tw, C)

            # Encode all video tile frames — temporal count stays the same as original
            tile_lat = vae.encode(tile_frames).to(device)  # (1, C, T, h_lat, w_lat)

            # Crop VACE conditioning spatially to the tile region.
            # Temporal counts are unchanged so phantom/reference frame math stays valid.
            pos_tile = self._crop_conditioning(positive, lat_y1, lat_x1, lat_y2, lat_x2)
            neg_tile = self._crop_conditioning(negative, lat_y1, lat_x1, lat_y2, lat_x2)

            latent_in = {"samples": tile_lat}

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

            # Decode tile
            tile_decoded = vae.decode(sampled["samples"])
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
