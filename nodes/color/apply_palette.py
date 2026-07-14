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
    (104, 14, 54),
    (207, 203, 85),
    (74, 66, 106),
    (85, 64, 54),
    (124, 85, 65),
    (45, 195, 205),
    (234, 165, 105),
    (135, 165, 154),
    (93, 56, 36),
    (174, 136, 32),
    (35, 35, 54),
    (105, 84, 75),
    (54, 55, 104),
    (84, 124, 146),
    (32, 25, 66),
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
    (56, 44, 74),
    (123, 206, 13),
    (133, 126, 124),
    (45, 16, 13),
    (152, 164, 105),
    (84, 45, 74),
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
    (76, 53, 16),
    (75, 104, 126),
    (174, 94, 76),
    (95, 75, 65),
    (195, 215, 225),
    (84, 94, 144),
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
    (164, 124, 95),
    (123, 115, 123),
    (185, 144, 105),
    (245, 154, 184),
    (115, 94, 75),
    (126, 83, 124),
    (156, 51, 75),
    (155, 115, 93),
    (104, 143, 36),
    (196, 103, 58),
    (57, 63, 113),
    (95, 153, 185),
    (95, 83, 124),
    (166, 123, 85),
    (105, 95, 104),
    (195, 143, 95),
    (206, 134, 85),
    (84, 54, 76),
    (224, 184, 144),
    (116, 83, 55),
    (164, 194, 205),
    (45, 73, 95),
    (236, 203, 125),
    (84, 114, 135),
    (116, 164, 183),
    (114, 77, 63),
    (66, 63, 64),
    (133, 75, 45),
    (164, 185, 174),
    (153, 77, 44),
    (184, 125, 84),
    (43, 66, 34),
    (234, 153, 95),
    (224, 187, 84),
    (225, 95, 144),
    (227, 101, 126),
    (94, 134, 155),
    (185, 36, 105),
    (153, 85, 54),
    (195, 93, 135),
    (205, 64, 115),
    (253, 49, 131),
    (64, 76, 115),
    (65, 114, 86),
    (146, 112, 27),
    (114, 85, 74),
    (66, 93, 36),
    (194, 146, 104),
    (174, 86, 55),
    (85, 66, 83),
    (217, 92, 124),
    (225, 115, 154),
    (53, 216, 222),
    (134, 65, 37),
    (64, 94, 116),
    (94, 185, 13),
    (184, 64, 94),
    (185, 95, 55),
    (46, 83, 65),
    (244, 136, 174),
    (136, 8, 31),
    (174, 126, 93),
    (105, 65, 35),
    (84, 133, 84),
    (175, 104, 64),
    (232, 164, 95),
    (164, 54, 53),
    (102, 227, 245),
    (195, 164, 145),
    (125, 95, 85),
    (244, 203, 146),
    (144, 125, 106),
    (145, 24, 75),
    (227, 36, 112),
    (83, 57, 52),
    (104, 65, 96),
    (126, 155, 143),
    (134, 174, 194),
    (64, 106, 83),
    (185, 133, 86),
    (253, 236, 243),
    (154, 94, 63),
    (175, 104, 126),
    (175, 135, 105),
    (76, 134, 135),
    (164, 27, 94),
    (76, 68, 111),
    (65, 105, 73),
    (124, 124, 165),
    (204, 166, 53),
    (146, 123, 85),
    (93, 47, 65),
    (55, 84, 95),
    (77, 71, 112),
    (64, 66, 94),
    (193, 15, 86),
    (97, 144, 134),
    (236, 164, 175),
    (136, 181, 37),
    (144, 114, 94),
    (36, 173, 187),
    (174, 206, 194),
    (226, 174, 32),
    (123, 7, 67),
    (64, 106, 142),
    (143, 145, 44),
    (187, 202, 124),
    (124, 85, 46),
    (52, 85, 57),
    (145, 54, 95),
    (125, 65, 44),
    (154, 105, 83),
    (76, 135, 104),
    (244, 236, 216),
    (134, 176, 164),
    (252, 146, 185),
    (126, 92, 66),
    (74, 104, 35),
    (236, 175, 114),
    (144, 135, 143),
    (134, 42, 25),
    (114, 114, 106),
    (145, 34, 44),
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
