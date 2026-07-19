from __future__ import annotations

import numpy as np
import torch
from numba import njit
import comfy.model_management
from comfy.utils import ProgressBar
from PIL import Image
from sklearn.cluster import KMeans

GAME_PALETTES: dict[str, np.ndarray] = {
    "256": np.array([
        (4, 4, 4),
        (13, 26, 53),
        (14, 85, 43),
        (34, 65, 94),
        (35, 52, 95),
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
        (63, 95, 95),
        (64, 24, 24),
        (64, 46, 43),
        (64, 63, 57),
        (64, 135, 134),
        (65, 13, 54),
        (65, 115, 144),
        (73, 55, 86),
        (74, 105, 64),
        (75, 54, 65),
        (75, 84, 64),
        (76, 85, 104),
        (76, 184, 65),
        (83, 36, 25),
        (83, 66, 103),
        (83, 146, 7),
        (84, 53, 37),
        (84, 56, 54),
        (84, 72, 106),
        (84, 74, 74),
        (85, 2, 3),
        (85, 55, 44),
        (85, 63, 55),
        (85, 86, 93),
        (86, 62, 46),
        (93, 25, 66),
        (93, 45, 82),
        (93, 46, 44),
        (93, 46, 56),
        (94, 53, 36),
        (94, 55, 44),
        (94, 65, 54),
        (94, 66, 63),
        (94, 75, 113),
        (95, 63, 46),
        (95, 74, 65),
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
        (105, 85, 122),
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
        (115, 54, 53),
        (115, 56, 43),
        (115, 83, 65),
        (115, 93, 76),
        (115, 95, 84),
        (115, 154, 175),
        (116, 1, 1),
        (116, 75, 93),
        (116, 82, 57),
        (123, 74, 75),
        (123, 76, 56),
        (124, 15, 35),
        (124, 64, 46),
        (124, 76, 63),
        (124, 82, 56),
        (124, 95, 34),
        (124, 96, 84),
        (124, 107, 103),
        (125, 6, 24),
        (125, 85, 64),
        (125, 93, 75),
        (125, 174, 13),
        (126, 103, 85),
        (134, 6, 27),
        (134, 85, 55),
        (134, 87, 65),
        (134, 96, 74),
        (134, 104, 85),
        (134, 106, 94),
        (134, 123, 116),
        (134, 145, 154),
        (135, 93, 66),
        (135, 114, 96),
        (135, 133, 134),
        (135, 164, 175),
        (136, 36, 114),
        (136, 65, 104),
        (136, 74, 95),
        (136, 102, 75),
        (136, 174, 193),
        (136, 182, 14),
        (143, 95, 66),
        (143, 116, 104),
        (144, 14, 36),
        (144, 85, 54),
        (144, 94, 96),
        (144, 95, 73),
        (144, 105, 84),
        (144, 114, 95),
        (145, 35, 45),
        (145, 36, 72),
        (145, 103, 75),
        (145, 124, 105),
        (146, 112, 86),
        (146, 113, 44),
        (147, 191, 7),
        (148, 231, 247),
        (153, 105, 75),
        (153, 116, 94),
        (153, 127, 114),
        (154, 103, 65),
        (154, 106, 83),
        (154, 183, 194),
        (155, 105, 146),
        (155, 113, 84),
        (155, 125, 104),
        (155, 134, 115),
        (155, 146, 144),
        (156, 73, 43),
        (156, 122, 95),
        (163, 126, 104),
        (164, 35, 75),
        (164, 116, 84),
        (164, 125, 95),
        (164, 135, 114),
        (164, 194, 25),
        (165, 94, 55),
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
        (174, 146, 124),
        (174, 186, 194),
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
        (185, 115, 134),
        (185, 124, 133),
        (185, 124, 144),
        (185, 125, 105),
        (185, 135, 94),
        (185, 153, 125),
        (185, 165, 144),
        (186, 116, 75),
        (186, 215, 222),
        (189, 116, 143),
        (194, 37, 84),
        (194, 65, 95),
        (194, 124, 85),
        (194, 125, 145),
        (194, 166, 143),
        (194, 174, 154),
        (195, 135, 75),
        (195, 135, 154),
        (195, 155, 114),
        (195, 156, 63),
        (195, 164, 135),
        (196, 215, 115),
        (202, 133, 161),
        (203, 134, 43),
        (204, 104, 157),
        (204, 134, 94),
        (204, 134, 147),
        (204, 134, 154),
        (204, 144, 115),
        (204, 144, 164),
        (204, 174, 145),
        (204, 185, 165),
        (204, 196, 194),
        (204, 244, 234),
        (205, 74, 103),
        (205, 103, 64),
        (205, 163, 64),
        (205, 182, 155),
        (206, 144, 156),
        (213, 195, 175),
        (214, 144, 156),
        (215, 144, 164),
        (215, 174, 84),
        (216, 155, 113),
        (223, 232, 176),
        (224, 75, 114),
        (224, 164, 85),
        (225, 164, 176),
        (225, 174, 125),
        (227, 93, 116),
        (227, 183, 133),
        (228, 228, 228),
        (233, 97, 122),
        (233, 222, 205),
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
        (250, 248, 253),
        (254, 144, 165),
        (254, 255, 255),
    ], dtype=np.uint8),
    "128": np.array([
        (4, 4, 4),
        (13, 26, 53),
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
        (84, 56, 54),
        (84, 72, 106),
        (84, 74, 74),
        (85, 2, 3),
        (85, 55, 44),
        (85, 63, 55),
        (86, 62, 46),
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
        (124, 82, 56),
        (124, 95, 34),
        (125, 6, 24),
        (125, 85, 64),
        (125, 93, 75),
        (134, 87, 65),
        (134, 96, 74),
        (134, 104, 85),
        (134, 123, 116),
        (135, 93, 66),
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
        (154, 183, 194),
        (155, 113, 84),
        (155, 125, 104),
        (155, 134, 115),
        (155, 146, 144),
        (156, 73, 43),
        (164, 125, 95),
        (164, 135, 114),
        (164, 194, 25),
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
        (185, 135, 94),
        (185, 165, 144),
        (186, 116, 75),
        (186, 215, 222),
        (189, 116, 143),
        (194, 65, 95),
        (194, 124, 85),
        (194, 125, 145),
        (194, 174, 154),
        (195, 135, 154),
        (195, 156, 63),
        (195, 164, 135),
        (202, 133, 161),
        (203, 134, 43),
        (204, 134, 154),
        (204, 174, 145),
        (204, 196, 194),
        (204, 244, 234),
        (205, 74, 103),
        (205, 163, 64),
        (206, 144, 156),
        (213, 195, 175),
        (215, 144, 164),
        (216, 155, 113),
        (224, 164, 85),
        (225, 174, 125),
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
        (13, 26, 53),
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
        (134, 104, 85),
        (135, 93, 66),
        (144, 14, 36),
        (155, 113, 84),
        (155, 134, 115),
        (155, 146, 144),
        (156, 73, 43),
        (164, 194, 25),
        (174, 136, 53),
        (175, 99, 128),
        (175, 155, 134),
        (183, 55, 87),
        (184, 173, 166),
        (185, 135, 94),
        (186, 116, 75),
        (186, 215, 222),
        (189, 116, 143),
        (194, 125, 145),
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
        (115, 83, 65),
        (115, 154, 175),
        (116, 1, 1),
        (124, 95, 34),
        (125, 85, 64),
        (134, 104, 85),
        (144, 14, 36),
        (155, 113, 84),
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
    rgb01 = rgb_u8.astype(np.float32) / 255.0
    return np.where(
        rgb01 <= 0.04045,
        rgb01 / 12.92,
        ((rgb01 + 0.055) / 1.055) ** 2.4,
    )


def _srgb_u8_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    # Matching in Lab (not raw/linear RGB) matters because hue lives entirely
    # in the a/b channels there; a linear-RGB distance can rate a hue-shifted
    # orange-brown chip "closer" to a rose/magenta skin-shadow tone than an
    # actual near-hue rose chip, since raw R/G/B can be numerically close
    # while looking nothing alike.
    linear = _srgb_u8_to_linear(rgb_u8).astype(np.float64)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883])
    scaled = xyz / white
    delta = 6 / 29

    fxyz = np.where(
        scaled > delta**3,
        np.cbrt(scaled),
        scaled / (3 * delta**2) + 4 / 29,
    )
    fx, fy, fz = fxyz[:, 0], fxyz[:, 1], fxyz[:, 2]
    lightness = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([lightness, a, b], axis=-1)


_NEAR_EXACT_LAB_THRESHOLD = 4.0


@njit(cache=True)
def _dither_lab_chunk(
    lab: np.ndarray,
    clean_lab: np.ndarray,
    palette_lab: np.ndarray,
    palette_u8: np.ndarray,
    near_exact_threshold: float,
    y_start: int,
    y_end: int,
    output: np.ndarray,
) -> None:
    height, width, _ = lab.shape
    num_colors = palette_lab.shape[0]

    for y in range(y_start, y_end):
        for x in range(width):
            cl0 = clean_lab[y, x, 0]
            cl1 = clean_lab[y, x, 1]
            cl2 = clean_lab[y, x, 2]

            source_best = 0
            source_best_dist = 1e18
            for c in range(num_colors):
                d0 = palette_lab[c, 0] - cl0
                d1 = palette_lab[c, 1] - cl1
                d2 = palette_lab[c, 2] - cl2
                dist = d0 * d0 + d1 * d1 + d2 * d2
                if dist < source_best_dist:
                    source_best_dist = dist
                    source_best = c

            if source_best_dist < near_exact_threshold:
                output[y, x, 0] = palette_u8[source_best, 0]
                output[y, x, 1] = palette_u8[source_best, 1]
                output[y, x, 2] = palette_u8[source_best, 2]
                continue

            w0 = lab[y, x, 0]
            w1 = lab[y, x, 1]
            w2 = lab[y, x, 2]

            best = 0
            best_dist = 1e18
            for c in range(num_colors):
                d0 = palette_lab[c, 0] - w0
                d1 = palette_lab[c, 1] - w1
                d2 = palette_lab[c, 2] - w2
                dist = d0 * d0 + d1 * d1 + d2 * d2
                if dist < best_dist:
                    best_dist = dist
                    best = c

            output[y, x, 0] = palette_u8[best, 0]
            output[y, x, 1] = palette_u8[best, 1]
            output[y, x, 2] = palette_u8[best, 2]

            e0 = w0 - palette_lab[best, 0]
            e1 = w1 - palette_lab[best, 1]
            e2 = w2 - palette_lab[best, 2]

            if x + 1 < width:
                lab[y, x + 1, 0] += e0 * (7 / 16)
                lab[y, x + 1, 1] += e1 * (7 / 16)
                lab[y, x + 1, 2] += e2 * (7 / 16)
            if y + 1 < height:
                if x - 1 >= 0:
                    lab[y + 1, x - 1, 0] += e0 * (3 / 16)
                    lab[y + 1, x - 1, 1] += e1 * (3 / 16)
                    lab[y + 1, x - 1, 2] += e2 * (3 / 16)
                lab[y + 1, x, 0] += e0 * (5 / 16)
                lab[y + 1, x, 1] += e1 * (5 / 16)
                lab[y + 1, x, 2] += e2 * (5 / 16)
                if x + 1 < width:
                    lab[y + 1, x + 1, 0] += e0 * (1 / 16)
                    lab[y + 1, x + 1, 1] += e1 * (1 / 16)
                    lab[y + 1, x + 1, 2] += e2 * (1 / 16)


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
    _OPAQUE_ALPHA_THRESHOLD = 127
    _PROGRESS_CHUNK_ROWS = 16

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

        # The game palette and the external swatch are the same for every
        # frame in the batch, so resolve them once up front instead of
        # redoing the work (or a disk read) per frame.
        fixed_palette = None
        if swatch == SWATCH_SOURCE_GAME_DEFAULT:
            fixed_palette = GAME_PALETTES[num_colors]
        elif swatch == SWATCH_SOURCE_EXTERNAL:
            fixed_palette = self._load_external_palette(swatch_path)

        total_frames = frames_np.shape[0]
        chunk_rows = self._PROGRESS_CHUNK_ROWS
        chunks_per_frame = max(1, (frames_np.shape[1] + chunk_rows - 1) // chunk_rows)
        total_progress_steps = total_frames * (chunks_per_frame + 1)
        pbar = ProgressBar(total_progress_steps)
        progress_value = 0

        def check_interrupted() -> None:
            comfy.model_management.throw_exception_if_processing_interrupted()

        def advance_progress(steps: int = 1) -> None:
            nonlocal progress_value
            progress_value = min(progress_value + steps, total_progress_steps)
            pbar.update_absolute(progress_value, total_progress_steps)

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
            palette = self._derive_palette(rgba, output_alpha, num_colors)
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
    def _derive_palette(cls, rgba: np.ndarray, alpha: np.ndarray, num_colors: int) -> np.ndarray:
        # Fit on opaque pixels only so transparent background doesn't skew
        # which colors get picked as the "best" ones in the image.
        pixels = rgba[:, :, :3].reshape(-1, 3)
        opaque = alpha.reshape(-1) > cls._OPAQUE_ALPHA_THRESHOLD
        fit_pixels = pixels[opaque] if np.any(opaque) else pixels
        return cls._kmeans_colors(fit_pixels, num_colors)

    @classmethod
    def _kmeans_colors(cls, colors: np.ndarray, num_colors: int) -> np.ndarray:
        kmeans = KMeans(
            n_clusters=min(num_colors, len(colors)),
            init="k-means++",
            max_iter=cls._KMEANS_MAX_ITERATIONS,
            tol=1e-3,
            random_state=42,
            n_init="auto",
        )
        kmeans.fit(colors.astype(np.float64))
        return np.clip(np.round(kmeans.cluster_centers_), 0, 255).astype(np.uint8)

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
        # blotch instead of a gradient. Floyd-Steinberg error diffusion in
        # Lab space breaks that up by carrying each pixel's quantization
        # error into its neighbors, so the region dithers between two close
        # chips instead of hard-cutting to one.
        #
        # Error diffusion has no natural stopping point: in a perfectly flat
        # region (e.g. a solid white background) the same tiny residual
        # keeps accumulating pixel after pixel until it finally crosses into
        # a completely different, unrelated palette chip (white -> pink),
        # which reads as random colored speckling in what should be a flat
        # fill. A source pixel that already lands within an imperceptible
        # distance of a palette entry is quantized hard against its own
        # (pre-diffusion) value and neither consumes incoming error nor
        # emits new error, which firewalls flat regions from ever
        # accumulating drift while leaving genuine gradients (which sit
        # meaningfully far from any single chip) dithered.
        #
        # The per-pixel loop is JIT-compiled (numba): each pixel depends on
        # the diffused error from its left/top neighbors, so it can't be
        # vectorized, but compiling it to native code avoids the per-pixel
        # numpy call overhead that dominates a plain Python loop -- roughly
        # 25x faster, which matters here since this runs per frame.
        palette_lab = _srgb_u8_to_lab(palette).astype(np.float64)
        palette_u8 = palette.astype(np.uint8)
        height, width, _ = image.shape
        clean_lab = _srgb_u8_to_lab(image.reshape(-1, 3)).reshape(height, width, 3).astype(np.float64)
        lab = clean_lab.copy()

        output = np.empty((height, width, 3), dtype=np.uint8)
        for y_start in range(0, height, chunk_rows):
            interrupt_callback()
            y_end = min(y_start + chunk_rows, height)
            _dither_lab_chunk(
                lab, clean_lab, palette_lab, palette_u8,
                _NEAR_EXACT_LAB_THRESHOLD, y_start, y_end, output,
            )
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
