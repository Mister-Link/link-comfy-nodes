from __future__ import annotations

import os
import uuid

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None  # type: ignore


class SpritesheetPreviewNode:
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_spritesheet"
    CATEGORY = "image/preview"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spritesheet": ("IMAGE",),
                "columns": (
                    "INT",
                    {
                        "default": 7,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "total_frames": (
                    "INT",
                    {
                        "default": 44,
                        "min": 1,
                        "max": 10000,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "fps": (
                    "INT",
                    {
                        "default": 12,
                        "min": 1,
                        "max": 60,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "scale": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "display": "number",
                    },
                ),
            },
        }

    def preview_spritesheet(
        self,
        spritesheet: torch.Tensor,
        columns: int,
        total_frames: int,
        fps: int,
        scale: int,
    ):
        if not isinstance(spritesheet, torch.Tensor):
            return {"ui": {"spritesheet_data": []}}

        sheet_np = spritesheet.cpu().numpy()

        if sheet_np.ndim == 4 and sheet_np.shape[0] > 0:
            sheet_np = sheet_np[0]
        elif sheet_np.ndim != 3:
            return {"ui": {"spritesheet_data": []}}

        sheet_255 = (np.clip(sheet_np, 0.0, 1.0) * 255).astype(np.uint8)
        pil_sheet = Image.fromarray(sheet_255)

        rows = (total_frames + columns - 1) // columns
        frame_width = pil_sheet.width // columns
        frame_height = pil_sheet.height // rows

        scaled_frame_width = frame_width * scale
        scaled_frame_height = frame_height * scale

        if scale > 1:
            new_width = pil_sheet.width * scale
            new_height = pil_sheet.height * scale
            pil_sheet = pil_sheet.resize(
                (new_width, new_height), Image.Resampling.NEAREST
            )

        if folder_paths:
            temp_dir = folder_paths.get_temp_directory()
        else:
            import tempfile

            temp_dir = tempfile.gettempdir()

        unique_id = uuid.uuid4().hex[:8]
        spritesheet_filename = f"spritesheet_static_{unique_id}.webp"
        spritesheet_filepath = os.path.join(temp_dir, spritesheet_filename)
        animation_filename = f"spritesheet_anim_{unique_id}.webp"
        animation_filepath = os.path.join(temp_dir, animation_filename)

        pil_sheet.save(spritesheet_filepath, format="WEBP", lossless=True)

        frames = []
        for i in range(total_frames):
            col = i % columns
            row = i // columns
            left = col * scaled_frame_width
            top = row * scaled_frame_height
            right = left + scaled_frame_width
            bottom = top + scaled_frame_height

            if right > pil_sheet.width or bottom > pil_sheet.height:
                print(f"Warning: Frame {i} exceeds spritesheet bounds. Skipping.")
                break

            frame = pil_sheet.crop((left, top, right, bottom))
            frames.append(frame)

        frame_duration = int(1000 / fps)
        if len(frames) > 0:
            durations = [frame_duration] * len(frames)
            frames[0].save(
                animation_filepath,
                format="WEBP",
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                lossless=False,
                quality=90,
                method=4,
            )

        return {
            "ui": {
                "spritesheet_data": [
                    {
                        "filename": spritesheet_filename,
                        "subfolder": "",
                        "type": "temp",
                        "width": scaled_frame_width,
                        "height": scaled_frame_height,
                        "frame_width": scaled_frame_width,
                        "frame_height": scaled_frame_height,
                        "columns": columns,
                        "total_frames": total_frames,
                        "fps": fps,
                        "animation_filename": animation_filename,
                    }
                ]
            }
        }
