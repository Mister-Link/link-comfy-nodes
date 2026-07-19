from __future__ import annotations

import math

import numpy as np
import torch
import comfy.model_management
from comfy.utils import ProgressBar
from PIL import Image

GAME_PALETTES: dict[str, np.ndarray] = {
    "256": np.array([
        (4, 4, 4),
        (13, 26, 53),
        (14, 85, 43),
        (15, 93, 185),
        (16, 73, 145),
        (34, 65, 94),
        (35, 52, 95),
        (35, 104, 183),
        (37, 82, 116),
        (44, 26, 24),
        (44, 61, 106),
        (44, 135, 35),
        (45, 45, 43),
        (45, 65, 44),
        (46, 43, 64),
        (46, 154, 34),
        (54, 35, 44),
        (54, 65, 72),
        (54, 94, 74),
        (57, 83, 53),
        (63, 65, 84),
        (63, 95, 95),
        (64, 24, 24),
        (64, 46, 43),
        (64, 63, 57),
        (64, 135, 134),
        (65, 13, 54),
        (65, 114, 144),
        (73, 55, 86),
        (74, 105, 64),
        (75, 54, 65),
        (75, 84, 64),
        (76, 63, 96),
        (76, 85, 104),
        (76, 184, 65),
        (83, 36, 25),
        (83, 66, 103),
        (83, 146, 7),
        (84, 53, 37),
        (84, 56, 54),
        (84, 72, 105),
        (84, 74, 74),
        (85, 2, 3),
        (85, 55, 44),
        (85, 63, 55),
        (86, 62, 46),
        (93, 25, 66),
        (93, 46, 44),
        (93, 46, 56),
        (94, 53, 36),
        (94, 55, 44),
        (94, 65, 54),
        (94, 65, 63),
        (94, 75, 113),
        (95, 63, 46),
        (95, 74, 65),
        (95, 136, 164),
        (96, 72, 55),
        (96, 83, 115),
        (102, 76, 25),
        (103, 56, 45),
        (103, 64, 46),
        (103, 155, 116),
        (104, 65, 54),
        (104, 75, 64),
        (104, 136, 154),
        (104, 163, 13),
        (105, 53, 36),
        (105, 64, 64),
        (105, 73, 55),
        (105, 84, 74),
        (105, 86, 123),
        (105, 95, 93),
        (105, 115, 84),
        (105, 123, 135),
        (105, 175, 163),
        (106, 33, 74),
        (106, 114, 153),
        (113, 65, 66),
        (113, 66, 54),
        (113, 206, 204),
        (114, 27, 56),
        (114, 46, 77),
        (114, 65, 44),
        (114, 75, 55),
        (114, 76, 63),
        (114, 86, 74),
        (115, 45, 54),
        (115, 56, 43),
        (115, 75, 93),
        (115, 83, 65),
        (115, 93, 76),
        (115, 95, 84),
        (115, 154, 175),
        (116, 1, 1),
        (116, 82, 57),
        (123, 74, 75),
        (123, 76, 56),
        (124, 15, 35),
        (124, 64, 46),
        (124, 76, 63),
        (124, 82, 56),
        (124, 95, 34),
        (124, 96, 84),
        (124, 106, 103),
        (125, 6, 24),
        (125, 85, 64),
        (125, 93, 75),
        (126, 103, 85),
        (134, 6, 27),
        (134, 85, 55),
        (134, 87, 65),
        (134, 96, 74),
        (134, 106, 94),
        (134, 123, 116),
        (134, 123, 145),
        (134, 145, 154),
        (135, 93, 66),
        (135, 104, 85),
        (135, 114, 96),
        (135, 133, 134),
        (135, 174, 193),
        (136, 36, 114),
        (136, 65, 104),
        (136, 74, 95),
        (136, 102, 75),
        (136, 182, 14),
        (143, 95, 66),
        (143, 116, 104),
        (144, 14, 36),
        (144, 85, 54),
        (144, 94, 96),
        (144, 95, 73),
        (144, 105, 84),
        (144, 114, 95),
        (145, 36, 72),
        (145, 103, 75),
        (145, 124, 105),
        (146, 112, 86),
        (146, 113, 44),
        (147, 191, 7),
        (148, 231, 247),
        (153, 116, 94),
        (154, 103, 65),
        (154, 105, 75),
        (154, 106, 83),
        (154, 127, 114),
        (154, 183, 194),
        (155, 114, 84),
        (155, 125, 104),
        (155, 134, 115),
        (155, 146, 144),
        (156, 73, 43),
        (156, 122, 95),
        (163, 126, 104),
        (164, 35, 75),
        (164, 115, 84),
        (164, 125, 95),
        (164, 194, 25),
        (165, 94, 55),
        (165, 135, 114),
        (165, 144, 124),
        (165, 164, 165),
        (166, 33, 63),
        (166, 94, 96),
        (166, 133, 106),
        (166, 155, 174),
        (173, 135, 106),
        (173, 137, 114),
        (173, 146, 134),
        (174, 114, 75),
        (174, 136, 53),
        (174, 145, 124),
        (175, 99, 128),
        (175, 104, 65),
        (175, 104, 125),
        (175, 114, 105),
        (175, 124, 85),
        (175, 143, 116),
        (175, 152, 126),
        (175, 155, 134),
        (176, 104, 35),
        (183, 55, 87),
        (183, 206, 75),
        (184, 145, 56),
        (184, 156, 134),
        (184, 162, 137),
        (184, 173, 166),
        (184, 195, 204),
        (185, 115, 134),
        (185, 124, 133),
        (185, 124, 144),
        (185, 125, 105),
        (185, 134, 94),
        (185, 153, 125),
        (185, 165, 144),
        (186, 116, 75),
        (186, 215, 222),
        (189, 116, 143),
        (193, 155, 125),
        (193, 174, 154),
        (194, 37, 84),
        (194, 65, 95),
        (194, 124, 85),
        (194, 125, 145),
        (194, 145, 115),
        (194, 163, 135),
        (194, 166, 144),
        (194, 172, 146),
        (195, 135, 75),
        (195, 135, 154),
        (195, 156, 63),
        (196, 215, 115),
        (202, 133, 161),
        (203, 134, 43),
        (204, 104, 157),
        (204, 134, 94),
        (204, 134, 154),
        (204, 144, 164),
        (204, 174, 145),
        (204, 176, 154),
        (204, 185, 165),
        (204, 196, 194),
        (204, 244, 234),
        (205, 74, 103),
        (205, 103, 64),
        (205, 163, 64),
        (205, 182, 155),
        (206, 144, 156),
        (214, 144, 156),
        (214, 195, 175),
        (215, 144, 164),
        (215, 174, 84),
        (216, 155, 113),
        (223, 232, 176),
        (224, 75, 114),
        (224, 164, 85),
        (225, 164, 176),
        (225, 174, 125),
        (225, 214, 205),
        (227, 93, 116),
        (227, 183, 133),
        (228, 228, 228),
        (233, 97, 122),
        (236, 53, 94),
        (236, 104, 125),
        (236, 114, 133),
        (244, 124, 145),
        (244, 194, 204),
        (245, 246, 246),
        (246, 105, 135),
        (246, 153, 165),
        (247, 115, 142),
        (248, 252, 253),
        (254, 144, 165),
        (254, 255, 255),
    ], dtype=np.uint8),
    "128": np.array([
        (4, 4, 4),
        (13, 26, 53),
        (15, 93, 185),
        (35, 52, 95),
        (37, 82, 116),
        (44, 26, 24),
        (44, 135, 35),
        (45, 45, 43),
        (45, 65, 44),
        (46, 43, 64),
        (54, 35, 44),
        (54, 65, 72),
        (63, 95, 95),
        (64, 24, 24),
        (64, 46, 43),
        (64, 63, 57),
        (65, 13, 54),
        (74, 105, 64),
        (75, 54, 65),
        (76, 184, 65),
        (83, 36, 25),
        (83, 66, 103),
        (84, 72, 105),
        (84, 74, 74),
        (85, 2, 3),
        (85, 55, 44),
        (85, 63, 55),
        (93, 25, 66),
        (93, 46, 56),
        (94, 55, 44),
        (94, 65, 54),
        (94, 75, 113),
        (95, 63, 46),
        (95, 74, 65),
        (103, 64, 46),
        (104, 65, 54),
        (104, 75, 64),
        (104, 163, 13),
        (105, 73, 55),
        (105, 84, 74),
        (105, 95, 93),
        (113, 206, 204),
        (114, 27, 56),
        (114, 75, 55),
        (114, 76, 63),
        (114, 86, 74),
        (115, 56, 43),
        (115, 83, 65),
        (115, 95, 84),
        (115, 154, 175),
        (116, 1, 1),
        (123, 74, 75),
        (124, 76, 63),
        (124, 95, 34),
        (125, 6, 24),
        (125, 85, 64),
        (125, 93, 75),
        (134, 87, 65),
        (134, 96, 74),
        (134, 123, 116),
        (135, 93, 66),
        (135, 104, 85),
        (135, 114, 96),
        (144, 14, 36),
        (144, 105, 84),
        (144, 114, 95),
        (145, 103, 75),
        (145, 124, 105),
        (146, 113, 44),
        (148, 231, 247),
        (153, 116, 94),
        (154, 103, 65),
        (154, 105, 75),
        (154, 183, 194),
        (155, 114, 84),
        (155, 125, 104),
        (155, 134, 115),
        (155, 146, 144),
        (156, 73, 43),
        (164, 115, 84),
        (164, 194, 25),
        (165, 135, 114),
        (165, 144, 124),
        (165, 164, 165),
        (174, 136, 53),
        (175, 99, 128),
        (175, 143, 116),
        (175, 155, 134),
        (183, 55, 87),
        (184, 145, 56),
        (184, 173, 166),
        (185, 125, 105),
        (185, 134, 94),
        (185, 165, 144),
        (186, 116, 75),
        (186, 215, 222),
        (189, 116, 143),
        (193, 174, 154),
        (194, 65, 95),
        (194, 124, 85),
        (194, 125, 145),
        (194, 163, 135),
        (195, 135, 154),
        (195, 156, 63),
        (202, 133, 161),
        (203, 134, 43),
        (204, 134, 154),
        (204, 174, 145),
        (204, 196, 194),
        (204, 244, 234),
        (205, 74, 103),
        (205, 163, 64),
        (206, 144, 156),
        (214, 195, 175),
        (215, 144, 164),
        (216, 155, 113),
        (224, 164, 85),
        (225, 174, 125),
        (225, 214, 205),
        (227, 93, 116),
        (228, 228, 228),
        (233, 97, 122),
        (236, 53, 94),
        (236, 104, 125),
        (244, 124, 145),
        (246, 105, 135),
        (254, 144, 165),
        (254, 255, 255),
    ], dtype=np.uint8),
    "64": np.array([
        (4, 4, 4),
        (15, 93, 185),
        (35, 52, 95),
        (37, 82, 116),
        (44, 26, 24),
        (54, 35, 44),
        (54, 65, 72),
        (64, 24, 24),
        (64, 46, 43),
        (65, 13, 54),
        (74, 105, 64),
        (75, 54, 65),
        (83, 66, 103),
        (85, 2, 3),
        (85, 55, 44),
        (85, 63, 55),
        (93, 25, 66),
        (94, 55, 44),
        (94, 65, 54),
        (95, 63, 46),
        (104, 75, 64),
        (104, 163, 13),
        (105, 73, 55),
        (105, 95, 93),
        (113, 206, 204),
        (114, 75, 55),
        (115, 56, 43),
        (115, 83, 65),
        (115, 154, 175),
        (116, 1, 1),
        (123, 74, 75),
        (124, 95, 34),
        (125, 85, 64),
        (134, 96, 74),
        (135, 93, 66),
        (135, 104, 85),
        (144, 14, 36),
        (155, 114, 84),
        (155, 134, 115),
        (155, 146, 144),
        (156, 73, 43),
        (164, 194, 25),
        (174, 136, 53),
        (175, 99, 128),
        (175, 143, 116),
        (175, 155, 134),
        (183, 55, 87),
        (184, 173, 166),
        (185, 134, 94),
        (186, 116, 75),
        (186, 215, 222),
        (189, 116, 143),
        (202, 133, 161),
        (204, 174, 145),
        (204, 244, 234),
        (205, 163, 64),
        (206, 144, 156),
        (225, 174, 125),
        (227, 93, 116),
        (228, 228, 228),
        (236, 53, 94),
        (236, 104, 125),
        (246, 105, 135),
        (254, 255, 255),
    ], dtype=np.uint8),
    "48": np.array([
        (4, 4, 4),
        (15, 93, 185),
        (35, 52, 95),
        (44, 26, 24),
        (54, 65, 72),
        (64, 24, 24),
        (64, 46, 43),
        (65, 13, 54),
        (74, 105, 64),
        (75, 54, 65),
        (83, 66, 103),
        (85, 2, 3),
        (85, 55, 44),
        (93, 25, 66),
        (94, 55, 44),
        (94, 65, 54),
        (104, 75, 64),
        (104, 163, 13),
        (105, 95, 93),
        (113, 206, 204),
        (114, 75, 55),
        (115, 154, 175),
        (116, 1, 1),
        (124, 95, 34),
        (125, 85, 64),
        (135, 104, 85),
        (144, 14, 36),
        (155, 114, 84),
        (155, 134, 115),
        (155, 146, 144),
        (156, 73, 43),
        (164, 194, 25),
        (174, 136, 53),
        (175, 99, 128),
        (175, 155, 134),
        (183, 55, 87),
        (184, 173, 166),
        (186, 116, 75),
        (189, 116, 143),
        (202, 133, 161),
        (204, 174, 145),
        (205, 163, 64),
        (225, 174, 125),
        (228, 228, 228),
        (236, 53, 94),
        (236, 104, 125),
        (246, 105, 135),
        (254, 255, 255),
    ], dtype=np.uint8),
}


def _srgb_u8_to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    rgb01 = rgb_u8.astype(np.float64) / 255.0
    return np.where(
        rgb01 <= 0.04045,
        rgb01 / 12.92,
        ((rgb01 + 0.055) / 1.055) ** 2.4,
    )


def _srgb_u8_to_oklab(rgb_u8: np.ndarray) -> np.ndarray:
    # Oklab (not CIELAB) because its a/b axes stay closer to perceptually
    # uniform hue/chroma across the whole lightness range, which matters
    # both for matching (a linear-RGB or even a naive Lab distance can rate
    # a hue-shifted chip "closer" than an actual near-hue one) and for
    # k-means, where a centroid is literally an average position in this
    # space -- a space where equal steps look like equal steps gives
    # centroids that better represent "the color halfway between".
    lin = _srgb_u8_to_linear(rgb_u8)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)

    lightness = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([lightness, a, b_], axis=-1)


# Squared-Oklab-distance floor below which a pixel is treated as an
# imperceptible match to its nearest palette entry and hard-snapped with no
# dithering at all. This (not a per-pixel "firewall" bolted onto an
# otherwise-unbounded diffusion process) is what keeps a genuinely flat or
# near-flat region -- a solid background, a flat fill -- perfectly flat in
# the output: see _map_colors_dithered for why the old Floyd-Steinberg
# approach could speckle a flat white background with unrelated colors.
_FLAT_MATCH_THRESHOLD = 0.0025

_BAYER_8 = np.array(
    [
        [0, 48, 12, 60, 3, 51, 15, 63],
        [32, 16, 44, 28, 35, 19, 47, 31],
        [8, 56, 4, 52, 11, 59, 7, 55],
        [40, 24, 36, 20, 43, 27, 39, 23],
        [2, 50, 14, 62, 1, 49, 13, 61],
        [34, 18, 46, 30, 33, 17, 45, 29],
        [10, 58, 6, 54, 9, 57, 5, 53],
        [42, 26, 38, 22, 41, 25, 37, 21],
    ],
    dtype=np.float64,
)
_BAYER_8_NORM = (_BAYER_8 + 0.5) / 64.0


NUM_COLORS_OPTIONS = ("256", "128", "64", "48")

SWATCH_SOURCE_GAME_DEFAULT = "Game Default"
SWATCH_SOURCE_EXTERNAL = "External Swatch"
SWATCH_SOURCE_FROM_INPUT = "From Input"
SWATCH_SOURCE_OPTIONS = (SWATCH_SOURCE_GAME_DEFAULT, SWATCH_SOURCE_EXTERNAL, SWATCH_SOURCE_FROM_INPUT)


class ApplyPaletteNode:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "apply_palette"
    CATEGORY = "color"

    _KMEANS_MAX_ITERATIONS = 100
    _KMEANS_PROGRESS_STAGES = 5
    _KMEANS_SEED = 42
    _OPAQUE_ALPHA_THRESHOLD = 127
    _PROGRESS_CHUNK_ROWS = 16
    _PALETTE_BUCKET_SIZE = 8
    # Fraction of the target palette size reserved for a dedicated
    # low-chroma cluster budget, separate from the chromatic budget. This is
    # the fix for "From Input" palettes that used to leave a frame's actual
    # background/near-neutral shades with no close match (they had to
    # compete for slots against saturated hues on equal footing) -- see
    # palette_forge.py for the full writeup.
    _NEUTRAL_BUDGET_FRACTION = 0.1875
    _NEUTRAL_CHROMA_THRESHOLD = 0.02

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "swatch": (SWATCH_SOURCE_OPTIONS, {"default": SWATCH_SOURCE_GAME_DEFAULT}),
                "num_colors": (NUM_COLORS_OPTIONS, {"default": NUM_COLORS_OPTIONS[0]}),
            },
            "optional": {
                "swatch_path": ("STRING", {"default": ""}),
                "alpha_opt": ("MASK",),
            },
        }

    def apply_palette(
        self,
        image: torch.Tensor,
        swatch: str,
        num_colors: str,
        swatch_path: str = "",
        alpha_opt: torch.Tensor | None = None,
    ):
        frames_np = (
            image.detach().cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        alpha_tensor = self._prepare_alpha(alpha_opt, image)
        num_colors_int = int(num_colors)

        total_frames = frames_np.shape[0]
        chunk_rows = self._PROGRESS_CHUNK_ROWS
        chunks_per_frame = max(1, (frames_np.shape[1] + chunk_rows - 1) // chunk_rows)
        # "From Input" replaces the single pre-dither tick with one tick per
        # KMeans refinement stage, since that fit is the slow, previously
        # invisible part of the run; the other two sources resolve their
        # palette near-instantly so a single tick is enough.
        palette_resolve_ticks = self._KMEANS_PROGRESS_STAGES if swatch == SWATCH_SOURCE_FROM_INPUT else 1
        setup_ticks = 1 if swatch == SWATCH_SOURCE_EXTERNAL else 0
        total_progress_steps = setup_ticks + total_frames * (chunks_per_frame + palette_resolve_ticks)
        pbar = ProgressBar(total_progress_steps)
        progress_value = 0

        def check_interrupted() -> None:
            comfy.model_management.throw_exception_if_processing_interrupted()

        def advance_progress(steps: int = 1) -> None:
            nonlocal progress_value
            progress_value = min(progress_value + steps, total_progress_steps)
            pbar.update_absolute(progress_value, total_progress_steps)

        # The game palette and the external swatch are the same for every
        # frame in the batch, so resolve them once up front instead of
        # redoing the work (or a disk read) per frame.
        fixed_palette = None
        if swatch == SWATCH_SOURCE_GAME_DEFAULT:
            fixed_palette = GAME_PALETTES[num_colors]
        elif swatch == SWATCH_SOURCE_EXTERNAL:
            fixed_palette = self._load_external_palette(swatch_path)
            advance_progress()

        palettized = []
        for idx in range(total_frames):
            check_interrupted()
            frame_alpha = None if alpha_tensor is None else alpha_tensor[idx].detach().cpu().numpy()
            palettized.append(
                self._apply_palette_to_frame(
                    frames_np[idx],
                    frame_alpha,
                    swatch,
                    fixed_palette,
                    num_colors_int,
                    progress_callback=advance_progress,
                    interrupt_callback=check_interrupted,
                )
            )

        result_tensor = torch.from_numpy(np.stack(palettized)).float() / 255.0
        return (result_tensor,)

    @staticmethod
    def _load_external_palette(swatch_path: str) -> np.ndarray:
        path = (swatch_path or "").strip()
        if not path:
            raise ValueError("swatch_path is required when swatch is 'External Swatch'.")

        swatch_image = np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)
        rgb = swatch_image[:, :, :3]
        visible = swatch_image[:, :, 3] > 0
        colors = rgb[visible] if np.any(visible) else rgb.reshape(-1, 3)
        if colors.size == 0:
            raise ValueError(f"Swatch image at {path!r} does not contain any visible palette colors.")
        return np.unique(colors, axis=0).astype(np.uint8)

    @staticmethod
    def _to_rgba(image: np.ndarray) -> np.ndarray:
        if image.shape[-1] == 4:
            return image
        alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate((image[:, :, :3], alpha), axis=2)

    @staticmethod
    def _flatten_over_white(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        rgb = image[:, :, :3].astype(np.float32)
        alpha_01 = alpha.astype(np.float32)[..., None] / 255.0
        white = np.full_like(rgb, 255.0)
        flattened = rgb * alpha_01 + white * (1.0 - alpha_01)
        return np.round(flattened).clip(0, 255).astype(np.uint8)

    def _apply_palette_to_frame(
        self,
        image: np.ndarray,
        alpha: np.ndarray | None,
        swatch: str,
        fixed_palette: np.ndarray | None,
        num_colors: int,
        progress_callback,
        interrupt_callback,
    ) -> np.ndarray:
        rgba = self._to_rgba(image)
        embedded_alpha = rgba[:, :, 3]
        output_alpha = embedded_alpha if alpha is None else np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
        flattened = self._flatten_over_white(rgba, output_alpha)

        if swatch == SWATCH_SOURCE_FROM_INPUT:
            palette = self._derive_palette(rgba, output_alpha, num_colors, progress_callback)
        else:
            palette = fixed_palette
            progress_callback()

        mapped = self._map_colors_dithered(
            flattened,
            palette,
            chunk_rows=self._PROGRESS_CHUNK_ROWS,
            progress_callback=progress_callback,
            interrupt_callback=interrupt_callback,
        )
        return np.dstack((mapped, output_alpha))

    @classmethod
    def _derive_palette(
        cls, rgba: np.ndarray, alpha: np.ndarray, num_colors: int, progress_callback
    ) -> np.ndarray:
        # Fit on opaque pixels only so transparent background doesn't skew
        # which colors get picked as the "best" ones in the image.
        pixels = rgba[:, :, :3].reshape(-1, 3)
        opaque = alpha.reshape(-1) > cls._OPAQUE_ALPHA_THRESHOLD
        fit_pixels = pixels[opaque] if np.any(opaque) else pixels
        return cls._kmeans_colors(fit_pixels, num_colors, progress_callback)

    @staticmethod
    def _bucketed_weighted_colors(rgb_u8_flat: np.ndarray, bucket_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_rgb[N,3] float, weight[N] float), bucketing to merge
        near-duplicate shades (grain, dither noise, JPEG-ish artifacts)
        before they compete for cluster budget individually. Uses
        np.bincount (a single C-level scatter-add per channel) so this stays
        fast even for a full-resolution frame's worth of pixels."""
        pixels = rgb_u8_flat.astype(np.int64)
        levels = -(-256 // bucket_size)  # ceil(256 / bucket_size)
        binned = pixels // bucket_size
        keys = (binned[:, 0] * levels + binned[:, 1]) * levels + binned[:, 2]
        num_keys = levels * levels * levels

        counts = np.bincount(keys, minlength=num_keys)
        sum_r = np.bincount(keys, weights=pixels[:, 0].astype(np.float64), minlength=num_keys)
        sum_g = np.bincount(keys, weights=pixels[:, 1].astype(np.float64), minlength=num_keys)
        sum_b = np.bincount(keys, weights=pixels[:, 2].astype(np.float64), minlength=num_keys)

        nonzero = counts > 0
        counts = counts[nonzero]
        means = np.stack([sum_r[nonzero], sum_g[nonzero], sum_b[nonzero]], axis=1) / counts[:, None]
        return means, counts.astype(np.float64)

    @staticmethod
    def _kmeanspp_init(points: np.ndarray, weights: np.ndarray, k: int, seed: int) -> np.ndarray:
        n = len(points)
        if k <= 0:
            return np.empty((0, 3))
        if n <= k:
            pad = k - n
            if pad == 0:
                return points.copy()
            rng = np.random.default_rng(seed)
            extra_idx = rng.choice(n, size=pad, replace=True) if n > 0 else np.array([], dtype=int)
            if n == 0:
                return np.zeros((k, 3))
            jitter = rng.normal(scale=1e-4, size=(pad, 3))
            return np.concatenate([points, points[extra_idx] + jitter], axis=0)

        rng = np.random.default_rng(seed)
        centers = np.empty((k, 3))
        probs = weights / weights.sum()
        centers[0] = points[rng.choice(n, p=probs)]
        closest_sq = np.sum((points - centers[0]) ** 2, axis=1)
        for i in range(1, k):
            weighted_d = closest_sq * weights
            total = weighted_d.sum()
            idx = rng.choice(n, p=weighted_d / total) if total > 0 else rng.choice(n)
            centers[i] = points[idx]
            new_d = np.sum((points - centers[i]) ** 2, axis=1)
            closest_sq = np.minimum(closest_sq, new_d)
        return centers

    @staticmethod
    def _kmeans_lloyd_iterations(
        points: np.ndarray, weights: np.ndarray, centers: np.ndarray, iters: int
    ) -> np.ndarray:
        if len(centers) == 0 or len(points) == 0:
            return centers
        for _ in range(iters):
            dists = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            assign = np.argmin(dists, axis=1)
            new_centers = centers.copy()
            for c in range(len(centers)):
                mask = assign == c
                if mask.any():
                    w = weights[mask]
                    new_centers[c] = (points[mask] * w[:, None]).sum(axis=0) / w.sum()
            if np.allclose(new_centers, centers, atol=1e-6):
                centers = new_centers
                break
            centers = new_centers
        return centers

    @staticmethod
    def _nearest_sampled_rgb(centers: np.ndarray, pool_rgb: np.ndarray, pool_oklab: np.ndarray) -> np.ndarray:
        # Snap each Oklab cluster center to the nearest *actual sampled*
        # sRGB color, rather than analytically inverting Oklab, guaranteeing
        # every palette entry genuinely occurs in the source frame.
        if len(centers) == 0:
            return np.empty((0, 3), dtype=np.uint8)
        if len(pool_rgb) == 0:
            return np.zeros((len(centers), 3), dtype=np.uint8)
        out = np.empty((len(centers), 3), dtype=np.uint8)
        for i, center in enumerate(centers):
            d = np.sum((pool_oklab - center) ** 2, axis=1)
            out[i] = pool_rgb[np.argmin(d)]
        return out

    @classmethod
    def _kmeans_colors(cls, colors: np.ndarray, num_colors: int, progress_callback) -> np.ndarray:
        # A single blocking fit over up to _KMEANS_MAX_ITERATIONS gives no
        # feedback until it's entirely done, which reads as a stalled
        # progress bar on any frame large enough for the fit to take a
        # while. Running the iteration budget in stages and calling
        # progress_callback once per stage (continuing from the previous
        # stage's centers) keeps the bar moving instead of going dark for
        # the whole fit.
        #
        # The palette is split into a dedicated neutral (low-chroma) budget
        # and a chromatic budget, each clustered independently in Oklab
        # space via weighted k-means. This is the actual fix for the
        # "dithering hallucinates on a flat background" failure: previously
        # the palette was one undifferentiated k-means fit in raw RGB, so a
        # frame's background shade only got a close match if it happened to
        # win enough slots against saturated colors on its own. Splitting
        # the budget guarantees near-white/near-black content always has
        # somewhere close to land.
        means, weights = cls._bucketed_weighted_colors(colors, cls._PALETTE_BUCKET_SIZE)
        oklab = _srgb_u8_to_oklab(means.astype(np.float64))

        # Cheap insurance: make true white/black *candidates* the neutral
        # cluster can snap to, without letting them dominate -- weight is
        # tiny relative to real content, so actual background shades win
        # wherever they exist.
        anchor_rgb = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.float64)
        anchor_weight = max(1.0, weights.sum() * 0.001)
        means = np.concatenate([means, anchor_rgb], axis=0)
        weights = np.concatenate([weights, np.full(2, anchor_weight)])
        oklab = np.concatenate([oklab, _srgb_u8_to_oklab(anchor_rgb)], axis=0)

        chroma = np.hypot(oklab[:, 1], oklab[:, 2])
        is_neutral = chroma < cls._NEUTRAL_CHROMA_THRESHOLD

        neutral_budget = min(num_colors, max(4, round(num_colors * cls._NEUTRAL_BUDGET_FRACTION)))
        chromatic_budget = max(0, num_colors - neutral_budget)

        neutral_points, neutral_weights = oklab[is_neutral], weights[is_neutral]
        chromatic_points, chromatic_weights = oklab[~is_neutral], weights[~is_neutral]

        neutral_centers = cls._kmeanspp_init(neutral_points, neutral_weights, neutral_budget, cls._KMEANS_SEED)
        chromatic_centers = cls._kmeanspp_init(
            chromatic_points, chromatic_weights, chromatic_budget, cls._KMEANS_SEED + 1
        )

        stages = max(1, cls._KMEANS_PROGRESS_STAGES)
        iters_per_stage = max(1, cls._KMEANS_MAX_ITERATIONS // stages)
        for _ in range(stages):
            neutral_centers = cls._kmeans_lloyd_iterations(
                neutral_points, neutral_weights, neutral_centers, iters_per_stage
            )
            chromatic_centers = cls._kmeans_lloyd_iterations(
                chromatic_points, chromatic_weights, chromatic_centers, iters_per_stage
            )
            progress_callback()

        neutral_rgb = cls._nearest_sampled_rgb(neutral_centers, means[is_neutral], neutral_points)
        chromatic_rgb = cls._nearest_sampled_rgb(chromatic_centers, means[~is_neutral], chromatic_points)

        combined = np.concatenate([neutral_rgb, chromatic_rgb], axis=0).astype(np.uint8)
        combined = np.unique(combined, axis=0)
        return combined[:num_colors]

    @staticmethod
    def _map_colors_dithered(
        image: np.ndarray,
        palette: np.ndarray,
        chunk_rows: int,
        progress_callback,
        interrupt_callback,
    ) -> np.ndarray:
        # Hard nearest-neighbor matching has no way to represent a source
        # shade that falls between two palette entries: a broad band of
        # similar midtones (e.g. a skin-shadow gradient) all snap to
        # whichever single chip is nearest, which reads as a flat, harsh
        # blotch instead of a gradient.
        #
        # The previous fix for that was Floyd-Steinberg error diffusion,
        # which has no natural floor: in a flat region the same tiny
        # quantization residue keeps accumulating pixel after pixel until it
        # finally crosses into a completely unrelated palette chip (white ->
        # pink speckling). The "near-exact firewall" that used to guard
        # against this only caught pixels that already happened to land
        # close to *some* palette entry -- it had no notion of whether the
        # surrounding region was actually flat, so a background shade that
        # sat just outside that radius still diffused freely and could run
        # away.
        #
        # This replaces diffusion with ordered (Bayer) dithering restricted
        # to each pixel's own two nearest palette colors. Every pixel's
        # choice depends only on a fixed spatial threshold pattern, never on
        # a neighbor's accumulated error -- there is nothing to accumulate,
        # so it is structurally impossible to land on a third, unrelated
        # hue. Pixels that are already an imperceptibly close match
        # (_FLAT_MATCH_THRESHOLD) hard-snap with no dithering, which is what
        # keeps a genuinely flat background perfectly flat.
        #
        # Because each pixel's decision no longer depends on its neighbors,
        # the whole frame resolves in one vectorized numpy pass -- no numba
        # JIT loop needed. The chunked loop below exists purely to keep the
        # progress bar and interrupt checks ticking at the same cadence as
        # before.
        height, width, _ = image.shape
        palette_u8 = palette.astype(np.uint8)
        palette_oklab = _srgb_u8_to_oklab(palette_u8.astype(np.float64))

        unique_pixels, inverse = np.unique(image.reshape(-1, 3), axis=0, return_inverse=True)
        unique_oklab = _srgb_u8_to_oklab(unique_pixels.astype(np.float64))

        dists = ((unique_oklab[:, None, :] - palette_oklab[None, :, :]) ** 2).sum(axis=2)
        if palette_oklab.shape[0] > 1:
            order = np.argsort(dists, axis=1)
            nearest_idx, second_idx = order[:, 0], order[:, 1]
        else:
            nearest_idx = np.zeros(len(unique_pixels), dtype=np.int64)
            second_idx = nearest_idx
        nearest_dist = np.take_along_axis(dists, nearest_idx[:, None], axis=1)[:, 0]
        second_dist = np.take_along_axis(dists, second_idx[:, None], axis=1)[:, 0]

        denom = np.maximum(nearest_dist + second_dist, 1e-12)
        mix_to_second = np.where(nearest_dist < _FLAT_MATCH_THRESHOLD, 0.0, nearest_dist / denom)

        px_nearest = nearest_idx[inverse].reshape(height, width)
        px_second = second_idx[inverse].reshape(height, width)
        px_mix = mix_to_second[inverse].reshape(height, width)

        bayer_tiled = np.tile(
            _BAYER_8_NORM, (math.ceil(height / 8), math.ceil(width / 8))
        )[:height, :width]

        output = np.empty((height, width, 3), dtype=np.uint8)
        for y_start in range(0, height, chunk_rows):
            interrupt_callback()
            y_end = min(y_start + chunk_rows, height)
            use_second = px_mix[y_start:y_end] > bayer_tiled[y_start:y_end]
            chosen = np.where(use_second, px_second[y_start:y_end], px_nearest[y_start:y_end])
            output[y_start:y_end] = palette_u8[chosen]
            progress_callback()

        return output

    @staticmethod
    def _prepare_alpha(
        alpha: torch.Tensor | None,
        frames: torch.Tensor,
    ) -> torch.Tensor | None:
        if alpha is None:
            return None
        alpha_tensor = alpha.detach().to(device=frames.device, dtype=frames.dtype)
        if alpha_tensor.ndim == 4 and alpha_tensor.shape[-1] == 1:
            alpha_tensor = alpha_tensor[..., 0]
        if alpha_tensor.ndim != 3:
            raise ValueError("Alpha must have shape (N, H, W) or (N, H, W, 1).")
        if alpha_tensor.shape[1:3] != frames.shape[1:3]:
            raise ValueError(
                f"Alpha size mismatch: frames={frames.shape[1:3]}, alpha={alpha_tensor.shape[1:3]}"
            )
        if alpha_tensor.shape[0] == 1 and frames.shape[0] > 1:
            alpha_tensor = alpha_tensor.expand(frames.shape[0], -1, -1)
        elif alpha_tensor.shape[0] != frames.shape[0]:
            raise ValueError(
                f"Alpha batch size must be 1 or match frame batch size: frames={frames.shape[0]}, alpha={alpha_tensor.shape[0]}"
            )
        return 1.0 - alpha_tensor.clamp(0.0, 1.0)
