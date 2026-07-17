from __future__ import annotations

import numpy as np
import torch

DEFAULT_GAME_PALETTE = np.array([
    (202, 133, 161),
    (189, 116, 143),
    (175, 99, 128),
    (254, 254, 255),
    (252, 68, 142),
    (163, 242, 5),
    (184, 145, 33),
    (64, 35, 56),
    (116, 1, 1),
    (134, 95, 74),
    (136, 177, 1),
    (74, 45, 35),
    (245, 125, 163),
    (124, 154, 174),
    (97, 244, 254),
    (83, 74, 115),
    (245, 206, 153),
    (88, 123, 45),
    (6, 4, 3),
    (44, 75, 54),
    (203, 113, 66),
    (196, 194, 194),
    (125, 94, 24),
    (184, 135, 94),
    (104, 65, 45),
    (236, 75, 125),
    (64, 104, 135),
    (52, 51, 124),
    (35, 24, 35),
    (253, 51, 132),
    (37, 51, 95),
    (85, 54, 34),
    (54, 36, 34),
    (204, 245, 234),
    (153, 146, 144),
    (144, 74, 44),
    (74, 0, 1),
    (225, 224, 225),
    (76, 172, 14),
    (167, 8, 67),
    (223, 96, 125),
    (222, 184, 74),
    (214, 166, 114),
    (94, 65, 54),
    (144, 13, 36),
    (73, 115, 75),
    (65, 25, 24),
    (104, 57, 73),
    (195, 155, 33),
    (47, 73, 135),
    (226, 145, 75),
    (55, 44, 55),
    (204, 23, 95),
    (194, 174, 155),
    (154, 84, 138),
    (175, 94, 55),
    (163, 115, 84),
    (115, 105, 104),
    (134, 13, 75),
    (125, 75, 54),
    (84, 55, 45),
    (64, 73, 125),
    (92, 67, 16),
    (95, 1, 3),
    (114, 84, 65),
    (115, 47, 42),
    (114, 73, 45),
    (144, 95, 65),
    (232, 37, 114),
    (163, 126, 31),
    (194, 66, 106),
    (184, 124, 146),
    (74, 54, 73),
    (255, 212, 32),
    (44, 63, 85),
    (62, 86, 35),
    (35, 23, 16),
    (176, 174, 173),
    (24, 45, 35),
    (104, 125, 115),
    (56, 94, 63),
    (154, 135, 115),
    (154, 95, 54),
    (252, 72, 143),
    (205, 44, 114),
    (133, 177, 37),
    (95, 63, 44),
    (114, 75, 54),
    (84, 75, 75),
    (103, 116, 83),
    (156, 202, 185),
    (94, 45, 34),
    (55, 84, 105),
    (252, 174, 205),
    (246, 213, 203),
    (207, 203, 85),
    (74, 66, 106),
    (85, 64, 54),
    (124, 85, 65),
    (45, 195, 205),
    (234, 165, 105),
    (135, 165, 154),
    (93, 56, 36),
    (174, 136, 32),
    (105, 84, 75),
    (54, 55, 104),
    (84, 124, 146),
    (115, 54, 35),
    (125, 83, 56),
    (174, 114, 75),
    (233, 147, 91),
    (245, 144, 176),
    (114, 85, 22),
    (84, 94, 86),
    (215, 86, 122),
    (95, 73, 55),
    (145, 103, 74),
    (66, 85, 123),
    (156, 77, 97),
    (251, 94, 155),
    (105, 74, 55),
    (164, 125, 104),
    (134, 5, 26),
    (204, 123, 76),
    (134, 182, 196),
    (135, 105, 95),
    (94, 55, 44),
    (125, 74, 45),
    (234, 185, 32),
    (245, 45, 126),
    (104, 85, 103),
    (43, 63, 114),
    (123, 206, 13),
    (133, 126, 124),
    (152, 164, 105),
    (187, 173, 73),
    (95, 63, 36),
    (75, 105, 43),
    (134, 85, 55),
    (216, 196, 135),
    (155, 83, 46),
    (154, 103, 65),
    (235, 224, 184),
    (144, 124, 114),
    (91, 116, 15),
    (186, 7, 81),
    (214, 195, 176),
    (195, 183, 175),
    (173, 197, 65),
    (115, 66, 44),
    (145, 105, 84),
    (175, 144, 115),
    (85, 155, 155),
    (125, 94, 74),
    (75, 104, 126),
    (174, 94, 76),
    (95, 75, 65),
    (195, 215, 225),
    (84, 94, 144),
    (95, 54, 85),
    (251, 113, 166),
    (198, 164, 184),
    (173, 154, 135),
    (135, 114, 95),
    (252, 205, 224),
    (154, 105, 75),
    (105, 84, 65),
    (65, 106, 104),
    (144, 84, 45),
    (104, 75, 64),
    (204, 155, 105),
    (135, 93, 65),
    (242, 194, 135),
    (175, 124, 85),
    (155, 235, 214),
    (244, 245, 154),
    (192, 125, 26),
    (75, 64, 75),
    (183, 105, 64),
    (143, 96, 74),
    (57, 94, 123),
    (104, 145, 165),
    (116, 144, 133),
    (42, 57, 104),
    (85, 63, 46),
    (222, 175, 125),
    (105, 66, 54),
    (196, 216, 203),
    (113, 25, 65),
    (114, 146, 176),
    (226, 76, 134),
    (165, 44, 93),
    (245, 65, 136),
    (204, 76, 114),
    (244, 132, 167),
    (153, 176, 194),
    (164, 156, 163),
    (135, 102, 26),
    (251, 83, 148),
    (163, 237, 6),
    (203, 163, 37),
    (75, 84, 135),
    (54, 94, 73),
    (133, 14, 65),
    (134, 83, 46),
    (145, 93, 55),
    (194, 164, 136),
    (133, 86, 64),
    (105, 54, 35),
    (194, 126, 84),
    (184, 164, 145),
    (63, 104, 67),
    (156, 113, 84),
    (165, 93, 55),
    (74, 115, 144),
    (174, 163, 155),
    (67, 56, 94),
    (93, 184, 175),
    (103, 76, 17),
    (95, 94, 93),
    (125, 66, 53),
    (144, 86, 54),
    (104, 44, 35),
    (225, 137, 83),
    (105, 74, 46),
    (254, 145, 194),
    (35, 84, 93),
    (145, 1, 25),
    (123, 54, 44),
    (194, 244, 84),
    (223, 34, 107),
    (135, 104, 84),
    (144, 75, 54),
    (54, 66, 63),
    (55, 86, 114),
    (184, 55, 104),
    (1, 164, 215),
    (165, 86, 53),
    (134, 76, 54),
    (125, 63, 37),
    (34, 134, 53),
    (165, 113, 75),
    (214, 154, 27),
    (252, 155, 195),
    (173, 136, 115),
    (164, 105, 65),
    (94, 156, 123),
    (94, 174, 205),
    (116, 132, 85),
    (125, 5, 26),
    (44, 75, 62),
    (205, 184, 165),
    (44, 67, 123),
    (144, 182, 44),
    (225, 55, 125),
    (251, 125, 173),
], dtype=np.uint8)


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


NUM_COLORS_OPTIONS = ("256",)


class ApplyPaletteNode:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("palettized_frames",)
    FUNCTION = "apply_palette"
    CATEGORY = "color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "use_swatch": ("BOOLEAN", {"default": True}),
                "num_colors": (NUM_COLORS_OPTIONS, {"default": NUM_COLORS_OPTIONS[0]}),
            },
            "optional": {
                "alpha_opt": ("MASK",),
            },
        }

    def apply_palette(
        self,
        frames: torch.Tensor,
        use_swatch: bool,
        num_colors: str,
        alpha_opt: torch.Tensor | None = None,
    ):
        frames_np = (
            frames.detach().cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        alpha_tensor = self._prepare_alpha(alpha_opt, frames)

        # Only "256" exists today; num_colors is kept as a dropdown so
        # future lower-color-count palettes can slot in here.
        palette = DEFAULT_GAME_PALETTE

        palettized = []
        for idx in range(frames_np.shape[0]):
            frame_alpha = None if alpha_tensor is None else alpha_tensor[idx].detach().cpu().numpy()
            palettized.append(
                self._apply_palette_to_frame(frames_np[idx], palette, frame_alpha, not use_swatch)
            )

        result_tensor = torch.from_numpy(np.stack(palettized)).float() / 255.0
        return (result_tensor,)

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
        palette: np.ndarray,
        alpha: np.ndarray | None,
        no_swatch: bool,
    ) -> np.ndarray:
        rgba = self._to_rgba(image)
        embedded_alpha = rgba[:, :, 3]
        output_alpha = embedded_alpha if alpha is None else np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
        flattened = self._flatten_over_white(rgba, output_alpha)
        mapped = flattened if no_swatch else self._map_colors_dithered(flattened, palette)
        return np.dstack((mapped, output_alpha))

    @staticmethod
    def _map_colors_dithered(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        # Hard nearest-neighbor matching has no way to represent a source
        # shade that falls between two palette entries: a broad band of
        # similar midtones (e.g. a skin-shadow gradient) all snap to
        # whichever single chip is nearest, which reads as a flat, harsh
        # blotch instead of a gradient. Floyd-Steinberg error diffusion in
        # Lab space breaks that up by carrying each pixel's quantization
        # error into its neighbors, so the region dithers between two close
        # chips instead of hard-cutting to one.
        palette_lab = _srgb_u8_to_lab(palette)
        height, width, _ = image.shape
        lab = _srgb_u8_to_lab(image.reshape(-1, 3)).reshape(height, width, 3).astype(np.float64)

        output = np.empty((height, width, 3), dtype=np.uint8)
        for y in range(height):
            row = lab[y]
            next_row = lab[y + 1] if y + 1 < height else None
            for x in range(width):
                working = row[x]
                distances = np.sum((palette_lab - working) ** 2, axis=-1)
                best = int(np.argmin(distances))
                output[y, x] = palette[best]
                error = working - palette_lab[best]

                if x + 1 < width:
                    row[x + 1] += error * (7 / 16)
                if next_row is not None:
                    if x - 1 >= 0:
                        next_row[x - 1] += error * (3 / 16)
                    next_row[x] += error * (5 / 16)
                    if x + 1 < width:
                        next_row[x + 1] += error * (1 / 16)

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
