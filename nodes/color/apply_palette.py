from __future__ import annotations

import numpy as np
import torch

DEFAULT_GAME_PALETTE = np.array([
    (254, 254, 255),
    (77, 69, 111),
    (191, 152, 32),
    (254, 255, 255),
    (12, 10, 14),
    (250, 146, 164),
    (103, 116, 81),
    (199, 104, 59),
    (245, 205, 150),
    (194, 160, 183),
    (254, 254, 254),
    (90, 65, 15),
    (255, 255, 255),
    (154, 84, 139),
    (30, 55, 39),
    (50, 212, 220),
    (194, 20, 85),
    (255, 195, 18),
    (205, 179, 161),
    (138, 105, 28),
    (70, 33, 30),
    (219, 90, 124),
    (255, 203, 108),
    (254, 215, 192),
    (180, 127, 82),
    (223, 146, 33),
    (120, 6, 57),
    (234, 146, 90),
    (156, 194, 178),
    (170, 167, 163),
    (153, 168, 104),
    (171, 97, 50),
    (138, 113, 107),
    (165, 128, 30),
    (89, 124, 151),
    (60, 31, 56),
    (247, 78, 22),
    (120, 54, 34),
    (223, 185, 71),
    (74, 119, 75),
    (175, 90, 102),
    (216, 170, 115),
    (215, 129, 120),
    (34, 22, 36),
    (166, 39, 92),
    (208, 122, 41),
    (95, 142, 131),
    (207, 126, 78),
    (79, 87, 83),
    (134, 180, 201),
    (113, 84, 20),
    (186, 70, 41),
    (121, 53, 64),
    (250, 230, 148),
    (128, 101, 146),
    (54, 94, 62),
    (38, 176, 190),
    (219, 95, 54),
    (232, 167, 98),
    (223, 194, 113),
    (52, 39, 36),
    (242, 198, 189),
    (99, 47, 85),
    (93, 61, 52),
    (138, 91, 50),
    (189, 153, 130),
    (122, 134, 121),
    (157, 107, 89),
    (203, 110, 90),
    (252, 212, 125),
    (103, 86, 104),
    (254, 238, 207),
    (252, 220, 239),
    (200, 66, 116),
    (145, 73, 38),
    (123, 135, 83),
    (36, 66, 80),
    (199, 196, 194),
    (239, 143, 27),
    (58, 91, 121),
    (225, 156, 144),
    (75, 52, 15),
    (240, 191, 131),
    (180, 69, 109),
    (204, 110, 121),
    (36, 25, 63),
    (123, 77, 60),
    (73, 51, 70),
    (151, 65, 84),
    (251, 239, 184),
    (36, 16, 14),
    (96, 45, 34),
    (134, 113, 137),
    (116, 155, 161),
    (99, 73, 16),
    (204, 164, 38),
    (228, 148, 76),
    (137, 163, 143),
    (45, 78, 53),
    (181, 142, 30),
    (225, 186, 84),
    (125, 94, 23),
    (160, 99, 67),
    (135, 162, 194),
    (86, 70, 67),
    (248, 149, 196),
    (224, 210, 160),
    (212, 250, 247),
    (230, 207, 139),
    (212, 182, 135),
    (142, 19, 74),
    (250, 122, 56),
    (198, 166, 143),
    (128, 121, 121),
    (197, 155, 161),
    (132, 66, 120),
    (81, 94, 71),
    (125, 136, 156),
    (132, 73, 46),
    (111, 92, 77),
    (180, 112, 67),
    (57, 45, 78),
    (52, 40, 54),
    (246, 211, 201),
    (102, 84, 130),
    (194, 69, 106),
    (217, 134, 41),
    (247, 131, 89),
    (197, 132, 84),
    (250, 181, 205),
    (232, 161, 105),
    (236, 156, 29),
    (234, 195, 159),
    (92, 106, 123),
    (233, 177, 165),
    (73, 39, 76),
    (248, 98, 56),
    (68, 57, 94),
    (173, 135, 31),
    (174, 130, 94),
    (182, 139, 159),
    (58, 36, 10),
    (132, 94, 82),
    (85, 44, 35),
    (254, 194, 65),
    (166, 183, 201),
    (181, 107, 51),
    (233, 194, 106),
    (228, 147, 144),
    (113, 139, 116),
    (116, 114, 100),
    (171, 126, 103),
    (255, 254, 255),
    (132, 151, 138),
    (179, 23, 80),
    (228, 154, 53),
    (182, 163, 157),
    (167, 87, 49),
    (246, 166, 134),
    (217, 126, 134),
    (152, 131, 116),
    (211, 118, 117),
    (166, 139, 113),
    (89, 82, 85),
    (55, 23, 17),
    (112, 156, 183),
    (251, 217, 96),
    (67, 226, 228),
    (37, 67, 47),
    (95, 121, 111),
    (184, 54, 103),
    (204, 115, 69),
    (30, 28, 33),
    (248, 217, 180),
    (250, 142, 78),
    (148, 144, 144),
    (183, 106, 68),
    (21, 42, 29),
    (216, 127, 75),
    (155, 119, 29),
    (62, 75, 59),
    (216, 169, 145),
    (205, 178, 192),
    (120, 59, 105),
    (231, 157, 88),
    (139, 20, 65),
    (212, 172, 60),
    (86, 70, 116),
    (219, 190, 188),
    (150, 86, 54),
    (226, 224, 226),
    (48, 26, 46),
    (224, 145, 91),
    (28, 11, 27),
    (190, 161, 195),
    (247, 177, 27),
    (196, 156, 33),
    (246, 179, 81),
    (48, 28, 26),
    (225, 136, 19),
    (194, 159, 48),
    (177, 106, 60),
    (62, 106, 68),
    (214, 94, 129),
    (224, 181, 134),
    (204, 81, 118),
    (209, 135, 84),
    (117, 60, 42),
    (190, 34, 92),
    (62, 17, 12),
    (117, 144, 157),
    (242, 198, 140),
    (134, 146, 99),
    (80, 42, 70),
    (165, 109, 80),
    (73, 47, 42),
    (238, 227, 187),
    (75, 68, 109),
    (204, 157, 105),
    (108, 102, 101),
    (154, 184, 186),
    (77, 104, 130),
    (222, 135, 133),
    (226, 196, 103),
    (157, 179, 161),
    (190, 139, 93),
    (136, 187, 169),
    (247, 118, 70),
    (250, 159, 188),
    (249, 103, 41),
    (135, 68, 38),
    (111, 85, 80),
    (90, 103, 96),
    (221, 176, 125),
    (213, 83, 118),
    (100, 138, 164),
    (188, 97, 55),
    (179, 99, 58),
    (181, 139, 113),
    (222, 137, 81),
    (145, 110, 26),
    (120, 100, 94),
    (179, 46, 103),
    (95, 85, 124),
    (73, 63, 102),
    (57, 49, 54),
    (252, 207, 67),
    (247, 144, 104),
    (187, 147, 30),
    (190, 183, 190),
    (42, 9, 9),
    (40, 23, 40),
    (109, 69, 56),
    (249, 194, 84),
    (117, 81, 71),
    (140, 54, 75),
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
