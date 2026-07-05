from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

import folder_paths  # type: ignore[import-untyped]
from comfy_execution.utils import get_executing_context


class PreviewImageAlpha:
    CATEGORY: str = "Video/Masking"
    RETURN_TYPES: tuple[str, ...] = ()
    FUNCTION: str = "preview_alpha"
    OUTPUT_NODE: bool = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "alpha": ("MASK",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def preview_alpha(
        self,
        frames: torch.Tensor,
        alpha: torch.Tensor,
        unique_id: str | None = None,
    ):
        frames_np = frames.cpu().numpy()
        alpha_np = alpha.cpu().numpy()

        if frames_np.shape[0] != alpha_np.shape[0]:
            raise ValueError(
                f"Frame count mismatch: frames={frames_np.shape[0]}, alpha={alpha_np.shape[0]}"
            )

        if frames_np.shape[1:3] != alpha_np.shape[1:3]:
            raise ValueError(
                f"Frame size mismatch: frames={frames_np.shape[1:3]}, alpha={alpha_np.shape[1:3]}"
            )

        # When frames/alpha arrive as a list (e.g. from a node with variable-sized
        # outputs), ComfyUI calls this node once per list item instead of once with
        # a full batch. Naming files by the intra-call frame index alone would then
        # collide across calls and each save would overwrite the last, so the node
        # index within the list is folded into the filename as well.
        executing_context = get_executing_context()
        list_index = executing_context.list_index if executing_context else None

        results = []

        for i in range(frames_np.shape[0]):
            rgba = np.concatenate([frames_np[i], alpha_np[i][:, :, np.newaxis]], axis=2)
            rgba_255 = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)
            pil_img = Image.fromarray(rgba_255, mode="RGBA")

            output_dir = folder_paths.get_output_directory()
            preview_name = f"preview_alpha_{unique_id or 'x'}_{list_index or 0:04d}_{i:04d}.png"
            preview_path = os.path.join(output_dir, preview_name)
            pil_img.save(preview_path)

            results.append(
                {"filename": preview_name, "subfolder": "", "type": "output"}
            )

        return {"ui": {"images": results}}
