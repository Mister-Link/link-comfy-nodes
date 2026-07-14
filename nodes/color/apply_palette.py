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
    (136, 177, 1),
    (134, 94, 74),
    (245, 125, 163),
    (124, 154, 174),
    (97, 244, 254),
    (44, 75, 54),
    (83, 74, 115),
    (245, 206, 153),
    (75, 43, 36),
    (88, 123, 45),
    (24, 14, 24),
    (203, 113, 66),
    (125, 94, 24),
    (196, 194, 194),
    (236, 75, 125),
    (64, 104, 135),
    (115, 144, 125),
    (52, 51, 124),
    (253, 51, 132),
    (37, 51, 95),
    (76, 172, 14),
    (167, 8, 67),
    (144, 74, 43),
    (153, 145, 144),
    (225, 224, 225),
    (223, 96, 125),
    (222, 184, 74),
    (73, 115, 75),
    (214, 166, 114),
    (104, 57, 73),
    (195, 155, 33),
    (55, 44, 55),
    (47, 73, 135),
    (226, 145, 75),
    (54, 36, 34),
    (85, 64, 54),
    (204, 23, 95),
    (154, 84, 138),
    (175, 94, 55),
    (85, 103, 95),
    (65, 25, 24),
    (134, 13, 75),
    (156, 202, 185),
    (64, 73, 125),
    (92, 67, 16),
    (84, 54, 45),
    (104, 44, 35),
    (232, 37, 114),
    (163, 126, 31),
    (184, 124, 146),
    (74, 54, 73),
    (255, 212, 32),
    (194, 66, 106),
    (44, 63, 85),
    (62, 86, 35),
    (115, 105, 104),
    (44, 23, 36),
    (24, 45, 35),
    (56, 94, 63),
    (103, 116, 83),
    (252, 72, 143),
    (176, 174, 173),
    (205, 44, 114),
    (133, 177, 37),
    (55, 84, 105),
    (85, 54, 33),
    (252, 174, 205),
    (104, 14, 54),
    (207, 203, 85),
    (74, 66, 106),
    (143, 154, 95),
    (35, 35, 54),
    (246, 213, 203),
    (75, 75, 74),
    (104, 125, 114),
    (45, 195, 205),
    (234, 165, 105),
    (174, 136, 32),
    (54, 55, 104),
    (84, 124, 146),
    (32, 25, 66),
    (126, 74, 54),
    (233, 147, 91),
    (245, 144, 176),
    (114, 85, 22),
    (154, 95, 54),
    (215, 86, 122),
    (66, 85, 123),
    (94, 64, 54),
    (156, 77, 97),
    (251, 94, 155),
    (105, 83, 75),
    (94, 54, 53),
    (204, 123, 76),
    (134, 182, 196),
    (136, 173, 156),
    (144, 124, 115),
    (35, 16, 14),
    (104, 85, 103),
    (234, 185, 32),
    (245, 45, 126),
    (43, 63, 114),
    (56, 44, 74),
    (123, 206, 13),
    (174, 243, 225),
    (115, 54, 35),
    (135, 104, 95),
    (187, 173, 73),
    (75, 105, 43),
    (114, 74, 56),
    (106, 133, 117),
    (216, 196, 135),
    (85, 155, 155),
    (73, 56, 53),
    (91, 116, 15),
    (186, 7, 81),
    (173, 197, 65),
    (155, 83, 45),
    (75, 104, 126),
    (133, 125, 125),
    (235, 223, 184),
    (124, 135, 85),
    (84, 94, 144),
    (95, 54, 85),
    (251, 113, 166),
    (195, 215, 225),
    (198, 164, 184),
    (94, 55, 44),
    (252, 205, 224),
    (74, 64, 75),
    (115, 74, 43),
    (242, 194, 135),
    (126, 155, 143),
    (244, 245, 154),
    (192, 125, 26),
    (194, 174, 155),
    (57, 94, 123),
    (94, 74, 56),
    (104, 145, 165),
    (74, 86, 82),
    (115, 94, 74),
    (42, 57, 104),
    (94, 85, 84),
    (183, 105, 64),
    (114, 146, 176),
    (226, 76, 134),
    (165, 44, 93),
    (245, 65, 136),
    (204, 76, 114),
    (244, 132, 167),
    (185, 135, 94),
    (122, 54, 44),
    (222, 175, 125),
    (135, 102, 26),
    (115, 83, 66),
    (251, 83, 148),
    (104, 66, 54),
    (163, 237, 6),
    (203, 163, 37),
    (75, 84, 135),
    (54, 94, 73),
    (133, 14, 65),
    (195, 183, 175),
    (83, 84, 65),
    (65, 106, 104),
    (63, 104, 67),
    (74, 115, 144),
    (67, 56, 94),
    (93, 184, 175),
    (103, 76, 17),
    (104, 74, 64),
    (225, 137, 83),
    (103, 146, 135),
    (254, 145, 194),
    (153, 176, 194),
    (35, 84, 93),
    (54, 64, 64),
    (124, 85, 66),
    (194, 244, 84),
    (95, 74, 65),
    (194, 126, 84),
    (134, 84, 46),
    (223, 34, 107),
    (55, 86, 114),
    (184, 55, 104),
    (204, 245, 234),
    (1, 164, 215),
    (34, 134, 53),
    (214, 154, 27),
    (252, 155, 195),
    (105, 124, 105),
    (94, 174, 205),
    (164, 155, 163),
    (94, 95, 84),
    (85, 145, 113),
    (44, 75, 62),
    (44, 67, 123),
    (144, 182, 44),
    (225, 55, 125),
    (251, 125, 173),
    (165, 93, 55),
    (154, 174, 95),
    (245, 154, 184),
    (126, 83, 124),
    (156, 51, 75),
    (104, 143, 36),
    (196, 103, 58),
    (57, 63, 113),
    (144, 75, 54),
    (95, 153, 185),
    (95, 83, 124),
    (84, 54, 76),
    (164, 194, 205),
    (84, 56, 52),
    (45, 73, 95),
    (236, 203, 125),
    (115, 65, 44),
    (84, 114, 135),
    (174, 156, 144),
    (116, 164, 183),
    (145, 104, 85),
    (144, 187, 171),
    (204, 155, 105),
    (115, 84, 74),
    (43, 66, 34),
    (234, 153, 95),
    (224, 187, 84),
    (225, 95, 144),
    (227, 101, 126),
    (94, 134, 155),
    (185, 36, 105),
    (195, 93, 135),
    (205, 64, 115),
    (105, 94, 104),
    (253, 49, 131),
    (64, 76, 115),
    (134, 163, 146),
    (65, 114, 86),
    (146, 112, 27),
    (116, 145, 133),
    (66, 93, 36),
    (86, 113, 103),
    (85, 66, 83),
    (217, 92, 124),
    (225, 115, 154),
    (144, 95, 75),
    (53, 216, 222),
    (64, 94, 116),
], dtype=np.uint8)


def _srgb_u8_to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    rgb01 = rgb_u8.astype(np.float32) / 255.0
    return np.where(
        rgb01 <= 0.04045,
        rgb01 / 12.92,
        ((rgb01 + 0.055) / 1.055) ** 2.4,
    )


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
            },
            "optional": {
                "swatch_opt": ("IMAGE",),
                "alpha_opt": ("MASK",),
            },
        }

    def apply_palette(
        self,
        frames: torch.Tensor,
        swatch_opt: torch.Tensor | None = None,
        alpha_opt: torch.Tensor | None = None,
    ):
        frames_np = (
            frames.detach().cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        alpha_tensor = self._prepare_alpha(alpha_opt, frames)

        if swatch_opt is None:
            palettes = [DEFAULT_GAME_PALETTE] * frames_np.shape[0]
        else:
            swatch_np = (
                swatch_opt.detach().cpu().numpy() * 255.0
            ).round().clip(0, 255).astype(np.uint8)
            if swatch_np.shape[0] not in (1, frames_np.shape[0]):
                raise ValueError("Swatch batch size must be 1 or match frame batch size.")
            palettes = [
                self._extract_palette(swatch_np[0 if swatch_np.shape[0] == 1 else idx])
                for idx in range(frames_np.shape[0])
            ]

        palettized = []
        for idx in range(frames_np.shape[0]):
            frame_alpha = None if alpha_tensor is None else alpha_tensor[idx].detach().cpu().numpy()
            palettized.append(self._apply_palette_to_frame(frames_np[idx], palettes[idx], frame_alpha))

        result_tensor = torch.from_numpy(np.stack(palettized)).float() / 255.0
        return (result_tensor,)

    @staticmethod
    def _extract_palette(image: np.ndarray) -> np.ndarray:
        rgba = ApplyPaletteNode._to_rgba(image)
        rgb = rgba[:, :, :3]
        visible = rgba[:, :, 3] > 0
        palette = rgb[visible] if np.any(visible) else rgb.reshape(-1, 3)
        if palette.size == 0:
            raise ValueError("Swatch image does not contain any visible palette colors.")
        return np.unique(palette, axis=0).astype(np.uint8)

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
    ) -> np.ndarray:
        rgba = self._to_rgba(image)
        embedded_alpha = rgba[:, :, 3]
        output_alpha = embedded_alpha if alpha is None else np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
        flattened = self._flatten_over_white(rgba, output_alpha)
        unique_pixels, inverse = np.unique(flattened.reshape(-1, 3), axis=0, return_inverse=True)
        mapped_unique = self._map_colors(unique_pixels, palette)
        mapped = mapped_unique[inverse].reshape(flattened.shape)
        return np.dstack((mapped, output_alpha))

    @staticmethod
    def _map_colors(colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
        colors_linear = _srgb_u8_to_linear(colors)
        palette_linear = _srgb_u8_to_linear(palette)
        delta = colors_linear[:, None, :] - palette_linear[None, :, :]
        distances = (
            (2.0 * delta[:, :, 0] ** 2)
            + (4.0 * delta[:, :, 1] ** 2)
            + (3.0 * delta[:, :, 2] ** 2)
        )
        return palette[np.argmin(distances, axis=1)]

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
