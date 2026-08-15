from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..color.apply_palette import NUM_COLORS_OPTIONS, _srgb_u8_to_oklab
from ..image.pixelate import ImagePixelateNode
from .node import ConvertToPixelArt
from .pixel_effect import PixelEffectModule


class PixelArtStudioNode:
    """One-stop pixel art: block reduction + palette + resize.

    Combines the three formerly separate stages:

    - Convert to Pixel Art's chunky block look (per-block winner-take-all
      color-family vote, kernel_size overlap, temporal hysteresis).
    - Palettize's global palette constraint, but fit once across ALL frames
      (k-means on the reduced block colors), so the palette cannot flicker
      frame to frame. Palette assignment additionally gets per-pixel
      temporal hysteresis: a pixel keeps its previous palette entry unless
      the new frame's color is clearly closer to a different one, which
      stops static details from popping a pixel thicker/thinner when their
      color drifts across a palette boundary.
    - Pixelate's size handling: optional width/height output resize using
      nearest-to-integer-multiple + box down, which keeps the pixel grid
      uniform at fractional scales.

    Alpha convention: embedded 4th channel in, weights votes/coverage,
    embedded 4th channel out (graded by alpha_threshold).
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pixel_art",)
    FUNCTION = "generate"
    CATEGORY = "image/transform"

    _model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "pixel_size": (
                    "INT",
                    {"default": 5, "min": 1, "max": 128, "step": 1},
                ),
                "kernel_size": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": (
                            "Sampling window per block. Equal to pixel_size = exact "
                            "tiling; larger overlaps neighboring blocks for a softer, "
                            "chunkier look."
                        ),
                    },
                ),
                "num_colors": (
                    NUM_COLORS_OPTIONS,
                    {
                        "default": NUM_COLORS_OPTIONS[0],
                        "tooltip": (
                            "Global palette size, fit once across all frames on the "
                            "reduced blocks (no per-frame palette flicker)."
                        ),
                    },
                ),
                "edge_style": (
                    ["hard", "soft"],
                    {
                        "default": "hard",
                        "tooltip": (
                            "hard = crisp pixel-art edges: the art grid is reduced by "
                            "nearest sampling (no invented in-between colors along "
                            "contours) and output alpha is snapped to fully opaque/"
                            "transparent, matching hand-drawn sprite edges. soft = "
                            "area-averaged reduce and graded alpha (antialiased look)."
                        ),
                    },
                ),
                "alpha_threshold": (
                    "FLOAT",
                    {
                        "default": 0.58,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Coverage level treated as ~50% opacity for the graded "
                            "output alpha. 0 = raw coverage."
                        ),
                    },
                ),
                "stability": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Temporal hysteresis on the block vote (video): previous "
                            "winner keeps a block unless clearly beaten. 0 disables."
                        ),
                    },
                ),
                "loop": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Treat the batch as a seamless loop: every temporal "
                            "mechanism (vote hysteresis, palette hysteresis, alpha "
                            "trigger) gets a hidden warm-up pass so frame 1 starts "
                            "from the end-of-sequence state -- no snap at the "
                            "last-to-first wrap. Disable for non-looping footage."
                        ),
                    },
                ),
            },
            "optional": {
                "width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Output width. 0 = input width. One of width/height set = keep aspect.",
                    },
                ),
                "height": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 1},
                ),
            },
        }

    @classmethod
    def _get_model(cls) -> PixelEffectModule:
        if cls._model is None:
            cls._model = PixelEffectModule()
            cls._model.eval()
        return cls._model

    @staticmethod
    def _fit_global_palette(
        block_rgb: np.ndarray, block_alpha: np.ndarray, num_colors: int
    ) -> np.ndarray:
        """K-means palette over the visible reduced blocks of all frames."""
        from sklearn.cluster import KMeans

        pixels = block_rgb.reshape(-1, 3)
        visible = block_alpha.reshape(-1) > 0.5
        fit = pixels[visible] if visible.any() else pixels
        # Blocks are already area averages; deduplicating keeps k-means fast
        # on large batches without changing what it sees materially.
        fit = np.unique(fit, axis=0)
        k = min(num_colors, len(fit))
        kmeans = KMeans(n_clusters=k, max_iter=100, tol=1e-3, random_state=42, n_init="auto")
        kmeans.fit(fit)
        return kmeans.cluster_centers_.round().clip(0, 255).astype(np.uint8)

    @staticmethod
    def _apply_palette_sequence(
        block_rgb: np.ndarray,
        palette_u8: np.ndarray,
        stability: float,
        loop: bool,
    ) -> np.ndarray:
        """Snap every frame's art pixels to the palette (oklab nearest),
        with per-pixel temporal hysteresis on the palette index.

        Even with the block vote stabilized upstream, the fractional
        box-reduce to the final art grid mixes slightly-drifting neighbor
        blocks, so a static pixel's color can wander across the midpoint
        between two palette entries -- the nearest entry then flips frame
        to frame, which reads as details popping a pixel thicker/thinner.
        Hysteresis: keep the previous frame's entry unless a different one
        is closer by a clear margin.
        """
        n, h, w, _ = block_rgb.shape
        palette_oklab = _srgb_u8_to_oklab(palette_u8.astype(np.float64))
        # Same semantics as the block-vote hysteresis: 1.2 * stability.
        # Measured on real footage: static-pixel palette flips drop from
        # ~7.5%/frame (no hysteresis) to ~3%/frame at margin 1.2.
        margin = 1.2 * stability
        eps = 1e-12

        prev_idx = None
        out = np.empty_like(block_rgb)
        # Loop mode: first pass only warms up prev_idx so frame 1 of the
        # kept pass continues from frame N -- no palette snap at the wrap.
        passes = 2 if (loop and margin > 0 and n > 1) else 1
        for _ in range(passes):
            for t in range(n):
                flat_oklab = _srgb_u8_to_oklab(
                    block_rgb[t].reshape(-1, 3).astype(np.float64)
                )
                dists = ((flat_oklab[:, None, :] - palette_oklab[None, :, :]) ** 2).sum(
                    axis=2
                )
                cand = dists.argmin(axis=1)
                if prev_idx is not None and margin > 0:
                    rows = np.arange(len(cand))
                    cand_d = dists[rows, cand]
                    inc_d = dists[rows, prev_idx]
                    keep = inc_d <= cand_d * (1.0 + margin) + eps
                    cand = np.where(keep, prev_idx, cand)
                prev_idx = cand
                out[t] = palette_u8[cand].reshape(h, w, 3)
        return out

    def generate(
        self,
        frames: torch.Tensor,
        pixel_size: int,
        kernel_size: int,
        num_colors: str,
        edge_style: str,
        alpha_threshold: float,
        stability: float = 1.0,
        loop: bool = True,
        width: int = 0,
        height: int = 0,
    ):
        process_device = ConvertToPixelArt._pick_processing_device(frames)
        images = frames.detach().to(device=process_device, dtype=torch.float32)
        if images.ndim != 4:
            raise ValueError("Expected frames with shape (N, H, W, C)")
        if images.numel():
            if float(images.max()) > 2.0:
                images = images / 255.0
            images = images.clamp(0.0, 1.0)

        has_alpha = images.shape[-1] == 4
        rgb = images[..., :3].permute(0, 3, 1, 2) * 255.0  # (N, 3, H, W)
        if has_alpha:
            alpha_norm = images[..., 3].unsqueeze(1)
        else:
            alpha_norm = torch.ones_like(rgb[:, :1])

        num_colors_int = int(num_colors)
        model = self._get_model()
        use_stabilize = images.shape[0] > 1 and stability > 0.0
        hysteresis_margin = 1.2 * stability if use_stabilize else 0.0
        vote_state = None
        prev_argmax = None

        # With loop enabled, a first hidden pass only warms up the
        # hysteresis state; outputs are kept from the second pass, whose
        # frame 1 then continues seamlessly from frame N.
        loop_state = loop and use_stabilize and images.shape[0] > 1
        block_rgbs = []
        block_alphas = []
        with torch.no_grad():
            for pass_idx in range(2 if loop_state else 1):
                final_pass = pass_idx == (1 if loop_state else 0)
                block_rgbs = []
                block_alphas = []
                for idx in range(images.shape[0]):
                    result_rgb, result_alpha, vote_state, prev_argmax = model(
                        rgb[idx : idx + 1],
                        alpha_norm[idx : idx + 1] * 255.0,
                        param_num_bins=32,
                        param_kernel_size=kernel_size,
                        param_pixel_size=pixel_size,
                        alpha_threshold=alpha_threshold,
                        vote_state=vote_state if use_stabilize else None,
                        prev_argmax=prev_argmax if use_stabilize else None,
                        hysteresis_margin=hysteresis_margin,
                    )
                    if not final_pass:
                        continue
                    # The module upsamples by pixel_size; every block is
                    # uniform, so sampling the top-left pixel of each block
                    # recovers the true block grid.
                    block_rgbs.append(
                        result_rgb[0, :, ::pixel_size, ::pixel_size]
                        .permute(1, 2, 0)
                        .clamp(0, 255)
                        .round()
                        .to(torch.uint8)
                        .cpu()
                        .numpy()
                    )
                    block_alphas.append(
                        (result_alpha[0, 0, ::pixel_size, ::pixel_size] / 255.0)
                        .clamp(0, 1)
                        .cpu()
                        .numpy()
                    )

        block_rgb = np.stack(block_rgbs)  # (N, hb, wb, 3) u8
        block_alpha = np.stack(block_alphas)  # (N, hb, wb) float

        # The final art grid is the smaller of the block grid and the
        # requested output size. If the target is smaller (e.g. sprite-size
        # output like 97x170 from a pixel_size-5 grid), reduce the grid
        # FIRST so the palette stage below constrains the actual final
        # pixels -- box-downscaling after palettization would re-mix
        # palette entries into unconstrained in-between colors.
        src_h, src_w = images.shape[1], images.shape[2]
        target = ImagePixelateNode._resolve_target_size((src_w, src_h), width, height)
        hb, wb = block_rgb.shape[1:3]
        art_w, art_h = min(wb, target[0]), min(hb, target[1])
        hard_edges = edge_style == "hard"
        # hard: nearest sampling -- every art pixel is a real block color,
        # no antialiased in-between colors along contours. soft: area
        # average (smoother, but edges blend).
        reduce_filter = Image.NEAREST if hard_edges else Image.BOX
        if (art_w, art_h) != (wb, hb):
            reduced_rgb = []
            reduced_alpha = []
            for i in range(block_rgb.shape[0]):
                reduced_rgb.append(
                    np.asarray(
                        Image.fromarray(block_rgb[i]).resize(
                            (art_w, art_h), reduce_filter
                        )
                    )
                )
                reduced_alpha.append(
                    np.asarray(
                        Image.fromarray(
                            (block_alpha[i] * 255.0).round().clip(0, 255).astype(np.uint8),
                            mode="L",
                        ).resize((art_w, art_h), reduce_filter),
                        dtype=np.float32,
                    )
                    / 255.0
                )
            block_rgb = np.stack(reduced_rgb)
            block_alpha = np.stack(reduced_alpha)

        if hard_edges:
            # Sprite-style binary alpha: an art pixel is either there or it
            # isn't, matching hand-drawn assets' solid silhouettes. The
            # threshold has temporal hysteresis (Schmitt trigger): a naive
            # >= 0.5 cut flips edge pixels opaque/transparent every time
            # their graded coverage drifts across the midpoint, which reads
            # as silhouettes pulsing thicker/thinner on static content. An
            # opaque pixel stays opaque until coverage clearly drops; a
            # transparent one stays until it clearly rises.
            band = 0.2 * stability
            binarized = np.empty_like(block_alpha)
            prev_state = None
            # Loop mode: warm-up pass seeds the trigger state so the
            # last-to-first wrap carries no snap.
            passes = 2 if (loop and band > 0 and block_alpha.shape[0] > 1) else 1
            for _ in range(passes):
                for t in range(block_alpha.shape[0]):
                    a = block_alpha[t]
                    if prev_state is None or band <= 0:
                        state = a >= 0.5
                    else:
                        state = np.where(prev_state, a >= 0.5 - band, a >= 0.5 + band)
                    binarized[t] = state
                    prev_state = state
            block_alpha = binarized.astype(np.float32)

        palette = self._fit_global_palette(block_rgb, block_alpha, num_colors_int)
        block_rgb = self._apply_palette_sequence(block_rgb, palette, stability, loop)

        # Resize the art grid to the output size (default: input frame size).
        out_rgb = []
        out_alpha = []
        for i in range(block_rgb.shape[0]):
            img = ImagePixelateNode._resize_pixel_art(
                Image.fromarray(block_rgb[i]), target, Image.BOX
            )
            out_rgb.append(np.asarray(img, dtype=np.float32) / 255.0)
            if has_alpha:
                a = ImagePixelateNode._resize_pixel_art(
                    Image.fromarray(
                        (block_alpha[i] * 255.0).round().clip(0, 255).astype(np.uint8),
                        mode="L",
                    ),
                    target,
                    Image.BOX,
                )
                a_np = np.asarray(a, dtype=np.float32) / 255.0
                if hard_edges:
                    # A fractional-scale resize can reintroduce partial
                    # alpha at edges; snap it back to binary.
                    a_np = (a_np >= 0.5).astype(np.float32)
                out_alpha.append(a_np)

        result = torch.from_numpy(np.stack(out_rgb))
        if has_alpha:
            result = torch.cat(
                [result, torch.from_numpy(np.stack(out_alpha)).unsqueeze(-1)], dim=-1
            )
        return (result.clamp(0.0, 1.0),)
