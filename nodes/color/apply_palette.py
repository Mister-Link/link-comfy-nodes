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
    (84, 54, 34),
    (136, 177, 1),
    (245, 125, 163),
    (124, 154, 174),
    (97, 244, 254),
    (83, 74, 115),
    (245, 206, 153),
    (134, 94, 74),
    (88, 123, 45),
    (44, 75, 54),
    (5, 4, 3),
    (203, 113, 66),
    (125, 94, 24),
    (196, 194, 194),
    (236, 75, 125),
    (64, 104, 135),
    (54, 36, 33),
    (52, 51, 124),
    (253, 51, 132),
    (37, 51, 95),
    (144, 74, 44),
    (74, 0, 1),
    (76, 172, 14),
    (153, 146, 144),
    (65, 25, 24),
    (167, 8, 67),
    (35, 24, 35),
    (225, 224, 225),
    (214, 166, 114),
    (223, 96, 125),
    (84, 54, 45),
    (222, 184, 74),
    (144, 13, 36),
    (73, 115, 75),
    (114, 73, 45),
    (104, 57, 73),
    (195, 155, 33),
    (47, 73, 135),
    (226, 145, 75),
    (35, 24, 15),
    (204, 23, 95),
    (55, 44, 55),
    (154, 84, 138),
    (115, 47, 42),
    (105, 95, 94),
    (175, 94, 55),
    (194, 174, 155),
    (134, 13, 75),
    (64, 73, 125),
    (94, 74, 55),
    (95, 1, 3),
    (54, 35, 17),
    (92, 67, 16),
    (104, 65, 44),
    (232, 37, 114),
    (154, 95, 54),
    (163, 126, 31),
    (194, 66, 106),
    (156, 202, 185),
    (184, 124, 146),
    (74, 54, 73),
    (255, 212, 32),
    (154, 135, 115),
    (44, 63, 85),
    (94, 64, 36),
    (62, 86, 35),
    (24, 45, 35),
    (104, 125, 115),
    (56, 94, 63),
    (85, 64, 54),
    (176, 174, 173),
    (252, 72, 143),
    (205, 44, 114),
    (133, 177, 37),
    (103, 116, 83),
    (185, 135, 94),
    (93, 56, 36),
    (125, 93, 55),
    (55, 84, 105),
    (252, 174, 205),
    (104, 14, 54),
    (207, 203, 85),
    (74, 66, 106),
    (246, 213, 203),
    (125, 75, 54),
    (135, 165, 154),
    (45, 195, 205),
    (234, 165, 105),
    (174, 136, 32),
    (54, 55, 104),
    (84, 124, 146),
    (115, 54, 35),
    (233, 147, 91),
    (94, 64, 54),
    (114, 84, 65),
    (245, 144, 176),
    (114, 85, 22),
    (215, 86, 122),
    (105, 84, 75),
    (66, 85, 123),
    (84, 94, 86),
    (156, 77, 97),
    (251, 94, 155),
    (145, 94, 65),
    (164, 125, 104),
    (95, 63, 44),
    (134, 5, 26),
    (204, 123, 76),
    (134, 182, 196),
    (234, 185, 32),
    (125, 74, 45),
    (245, 45, 126),
    (43, 63, 114),
    (84, 75, 75),
    (135, 114, 105),
    (123, 206, 13),
    (126, 124, 123),
    (174, 243, 225),
    (94, 55, 44),
    (235, 224, 185),
    (104, 85, 103),
    (114, 74, 55),
    (152, 164, 105),
    (187, 173, 73),
    (174, 114, 75),
    (75, 105, 43),
    (105, 75, 44),
    (216, 196, 135),
    (155, 83, 46),
    (84, 65, 45),
    (91, 116, 15),
    (186, 7, 81),
    (195, 183, 175),
    (164, 115, 84),
    (173, 197, 65),
    (115, 66, 44),
    (85, 155, 155),
    (75, 104, 126),
    (174, 94, 76),
    (124, 85, 65),
    (213, 195, 175),
    (84, 94, 144),
    (95, 54, 85),
    (251, 113, 166),
    (195, 215, 225),
    (198, 164, 184),
    (134, 83, 46),
    (154, 114, 66),
    (252, 205, 224),
    (196, 216, 203),
    (65, 106, 104),
    (144, 84, 45),
    (95, 72, 45),
    (242, 194, 135),
    (73, 56, 53),
    (104, 84, 65),
    (116, 144, 133),
    (244, 245, 154),
    (192, 125, 26),
    (135, 84, 55),
    (183, 105, 64),
    (57, 94, 123),
    (173, 154, 134),
    (104, 145, 165),
    (42, 57, 104),
    (75, 64, 75),
    (134, 115, 95),
    (222, 175, 125),
    (85, 62, 35),
    (105, 66, 54),
    (114, 146, 176),
    (226, 76, 134),
    (165, 44, 93),
    (245, 65, 136),
    (204, 76, 114),
    (244, 132, 167),
    (155, 103, 65),
    (95, 75, 65),
    (204, 155, 105),
    (135, 102, 26),
    (144, 95, 75),
    (251, 83, 148),
    (163, 237, 6),
    (203, 163, 37),
    (75, 84, 135),
    (54, 94, 73),
    (133, 14, 65),
    (175, 145, 115),
    (145, 93, 55),
    (114, 83, 46),
    (145, 104, 84),
    (105, 54, 35),
    (125, 94, 85),
    (194, 126, 84),
    (104, 74, 55),
    (63, 104, 67),
    (165, 93, 55),
    (74, 115, 144),
    (67, 56, 94),
    (93, 184, 175),
    (103, 76, 17),
    (125, 83, 55),
    (125, 66, 53),
    (144, 86, 54),
    (133, 86, 64),
    (104, 44, 35),
    (225, 137, 83),
    (254, 145, 194),
    (153, 176, 194),
    (35, 84, 93),
    (145, 1, 25),
    (146, 123, 85),
    (123, 54, 44),
    (194, 244, 84),
    (223, 34, 107),
    (144, 75, 54),
    (54, 66, 63),
    (55, 86, 114),
    (184, 55, 104),
    (204, 245, 234),
    (1, 164, 215),
    (83, 54, 27),
    (165, 86, 53),
    (134, 76, 54),
    (164, 155, 163),
    (125, 63, 37),
    (34, 134, 53),
    (214, 154, 27),
    (252, 155, 195),
    (164, 105, 65),
    (115, 94, 74),
    (94, 156, 123),
    (94, 174, 205),
    (116, 132, 85),
    (125, 5, 26),
    (44, 75, 62),
    (44, 67, 123),
    (134, 104, 83),
    (144, 182, 44),
    (114, 105, 113),
    (225, 55, 125),
    (251, 125, 173),
    (245, 154, 184),
    (163, 145, 135),
    (126, 83, 124),
    (156, 51, 75),
    (104, 143, 36),
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
