from __future__ import annotations

import numpy as np
import torch

from .palette_transfer import detect_background_color, reinhard_transfer_lab


class MatchColorPaletteNode:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graded_images",)
    FUNCTION = "match_color_palette"
    CATEGORY = "color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
            },
        }

    def match_color_palette(
        self,
        image_ref: torch.Tensor,
        image_target: torch.Tensor,
    ):
        ref_np = (
            image_ref.detach().cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        tgt_np = (
            image_target.detach().cpu().numpy() * 255.0
        ).round().clip(0, 255).astype(np.uint8)

        if ref_np.shape[0] not in (1, tgt_np.shape[0]):
            raise ValueError(
                "Reference batch size must be 1 or match target batch size."
            )

        results = []
        for idx in range(tgt_np.shape[0]):
            ref_idx = 0 if ref_np.shape[0] == 1 else idx
            ref_rgb, ref_alpha = self._split_alpha(ref_np[ref_idx])
            tgt_rgb, tgt_alpha = self._split_alpha(tgt_np[idx])

            reference_mask = ref_alpha > 0 if ref_alpha is not None else None
            target_mask_alpha = tgt_alpha > 0 if tgt_alpha is not None else None

            bg_rgb = detect_background_color(tgt_rgb)
            if reference_mask is not None:
                reference_content_mask = reference_mask & np.any(
                    ref_rgb != bg_rgb, axis=-1
                )
            else:
                reference_content_mask = np.any(ref_rgb != bg_rgb, axis=-1)
            target_content_mask = np.any(tgt_rgb != bg_rgb, axis=-1)
            if target_mask_alpha is not None:
                target_content_mask &= target_mask_alpha

            graded_rgb = reinhard_transfer_lab(
                tgt_rgb,
                ref_rgb,
                target_mask=target_content_mask,
                reference_mask=reference_content_mask,
            )

            if tgt_alpha is not None:
                graded = np.dstack((graded_rgb, tgt_alpha))
            else:
                graded = graded_rgb
            results.append(graded.astype(np.uint8))

        result_tensor = torch.from_numpy(np.stack(results)).float() / 255.0
        return (result_tensor,)

    @staticmethod
    def _split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if image.shape[-1] == 4:
            return image[:, :, :3], image[:, :, 3]
        return image[:, :, :3], None
