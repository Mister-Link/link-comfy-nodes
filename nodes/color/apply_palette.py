from __future__ import annotations

import numpy as np
import torch

DEFAULT_GAME_PALETTE = np.array([
    (202, 133, 161),
    (189, 116, 143),
    (175, 99, 128),
    (254, 254, 255),
    (77, 69, 111),
    (196, 156, 33),
    (254, 255, 255),
    (12, 10, 14),
    (103, 116, 81),
    (199, 104, 59),
    (245, 205, 150),
    (254, 254, 254),
    (90, 65, 15),
    (255, 255, 255),
    (170, 167, 163),
    (30, 55, 39),
    (194, 160, 183),
    (218, 88, 122),
    (255, 195, 18),
    (250, 146, 164),
    (254, 215, 192),
    (138, 105, 28),
    (154, 84, 139),
    (70, 33, 30),
    (255, 203, 108),
    (180, 127, 82),
    (223, 146, 33),
    (234, 146, 90),
    (198, 166, 143),
    (168, 130, 32),
    (153, 168, 104),
    (171, 97, 50),
    (157, 29, 85),
    (138, 113, 107),
    (60, 31, 56),
    (74, 119, 75),
    (156, 194, 178),
    (194, 20, 85),
    (216, 170, 115),
    (223, 185, 71),
    (215, 129, 120),
    (34, 22, 36),
    (208, 122, 41),
    (207, 126, 78),
    (79, 87, 83),
    (113, 84, 20),
    (186, 70, 41),
    (250, 230, 148),
    (151, 65, 84),
    (122, 134, 121),
    (54, 94, 62),
    (219, 95, 54),
    (232, 167, 98),
    (180, 69, 109),
    (223, 194, 113),
    (52, 39, 36),
    (242, 198, 189),
    (93, 61, 52),
    (204, 110, 121),
    (157, 107, 89),
    (203, 110, 90),
    (181, 142, 30),
    (252, 212, 125),
    (254, 238, 207),
    (252, 220, 239),
    (150, 170, 191),
    (150, 86, 54),
    (205, 179, 161),
    (199, 196, 194),
    (239, 143, 27),
    (225, 156, 144),
    (75, 52, 15),
    (200, 66, 116),
    (240, 191, 131),
    (73, 51, 70),
    (251, 239, 184),
    (181, 139, 113),
    (36, 16, 14),
    (175, 90, 102),
    (99, 73, 16),
    (228, 148, 76),
    (45, 78, 53),
    (225, 186, 84),
    (125, 94, 23),
    (86, 70, 67),
    (152, 76, 41),
    (224, 210, 160),
    (212, 250, 247),
    (230, 207, 139),
    (212, 182, 135),
    (179, 46, 103),
    (128, 121, 121),
    (197, 155, 161),
    (85, 44, 35),
    (180, 112, 67),
    (119, 99, 88),
    (52, 40, 54),
    (148, 168, 152),
    (246, 211, 201),
    (194, 69, 106),
    (217, 134, 41),
    (197, 132, 84),
    (232, 161, 105),
    (236, 156, 29),
    (165, 109, 80),
    (234, 195, 159),
    (233, 177, 165),
    (67, 57, 94),
    (182, 139, 159),
    (132, 151, 138),
    (174, 130, 94),
    (155, 119, 29),
    (181, 107, 51),
    (233, 194, 106),
    (228, 147, 144),
    (116, 114, 100),
    (255, 254, 255),
    (179, 23, 80),
    (206, 169, 54),
    (228, 154, 53),
    (182, 163, 157),
    (217, 126, 134),
    (152, 131, 116),
    (189, 153, 130),
    (211, 118, 117),
    (89, 82, 85),
    (55, 23, 17),
    (204, 164, 38),
    (37, 67, 47),
    (204, 115, 69),
    (30, 28, 33),
    (214, 94, 129),
    (167, 87, 49),
    (248, 217, 180),
    (148, 144, 144),
    (183, 106, 68),
    (21, 42, 29),
    (216, 127, 75),
    (166, 121, 96),
    (216, 169, 145),
    (173, 135, 31),
    (166, 139, 113),
    (205, 178, 192),
    (231, 157, 88),
    (187, 148, 31),
    (86, 70, 116),
    (219, 190, 188),
    (226, 224, 226),
    (48, 26, 46),
    (224, 145, 91),
    (28, 11, 27),
    (190, 161, 195),
    (48, 28, 26),
    (225, 136, 19),
    (160, 99, 67),
    (177, 106, 60),
    (154, 184, 186),
    (224, 181, 134),
    (204, 81, 118),
    (209, 135, 84),
    (190, 34, 92),
    (242, 198, 140),
    (194, 159, 48),
    (191, 152, 32),
    (238, 227, 187),
    (75, 68, 109),
    (204, 157, 105),
    (108, 102, 101),
    (183, 59, 101),
    (222, 135, 133),
    (226, 196, 103),
    (190, 139, 93),
    (90, 103, 96),
    (221, 176, 125),
    (70, 54, 50),
    (188, 97, 55),
    (179, 99, 58),
    (222, 137, 81),
    (73, 63, 102),
    (57, 49, 54),
    (100, 85, 80),
    (190, 183, 190),
    (40, 23, 40),
    (133, 100, 26),
    (74, 41, 35),
    (200, 163, 160),
    (185, 140, 99),
    (82, 74, 115),
    (253, 206, 116),
    (185, 150, 40),
    (66, 35, 56),
    (182, 217, 224),
    (189, 111, 68),
    (170, 160, 170),
    (210, 79, 116),
    (228, 209, 152),
    (171, 94, 43),
    (223, 148, 135),
    (193, 128, 86),
    (62, 24, 21),
    (184, 54, 103),
    (166, 183, 201),
    (194, 153, 171),
    (158, 84, 44),
    (229, 141, 85),
    (246, 207, 155),
    (199, 163, 185),
    (78, 70, 111),
    (237, 230, 204),
    (236, 174, 107),
    (239, 187, 179),
    (45, 32, 47),
    (162, 124, 30),
    (166, 39, 92),
    (178, 158, 144),
    (77, 70, 112),
    (161, 206, 189),
    (234, 151, 95),
    (29, 18, 23),
    (98, 71, 19),
    (34, 62, 42),
    (213, 85, 120),
    (221, 95, 128),
    (219, 182, 74),
    (158, 139, 134),
    (27, 50, 35),
    (205, 175, 154),
    (194, 182, 172),
    (185, 148, 171),
    (223, 201, 129),
    (228, 216, 174),
    (76, 68, 110),
    (213, 161, 108),
    (178, 144, 121),
    (201, 161, 38),
    (230, 187, 135),
    (247, 208, 200),
    (203, 110, 64),
    (157, 179, 161),
    (221, 191, 145),
    (226, 189, 81),
    (160, 153, 143),
    (78, 70, 112),
    (232, 162, 92),
    (73, 65, 105),
    (232, 170, 158),
    (219, 172, 120),
    (193, 103, 59),
    (184, 145, 30),
    (203, 76, 111),
    (245, 205, 152),
    (189, 115, 70),
    (221, 140, 136),
    (66, 34, 30),
    (196, 162, 184),
    (229, 151, 82),
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
