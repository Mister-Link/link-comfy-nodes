"""Simple Video Preview Node - A streamlined video preview node with minimal inputs."""

import os
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch
from PIL import Image

import folder_paths  # type: ignore[import-untyped]


def tensor_to_pil(tensor: torch.Tensor) -> list[Image.Image]:
    """Convert a tensor batch to PIL images."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    images = []
    for img_tensor in tensor:
        img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(img_array))

    return images


class PreviewAnimation:
    """A simple video preview node that just takes fps and frames."""

    CATEGORY = "Video/Preview"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "preview_animation"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.01, "max": 120.0, "step": 0.01},
                ),
            },
        }

    def preview_animation(
        self,
        frames: torch.Tensor,
        fps: float,
    ) -> dict[str, Any]:
        """Generate a preview animation from the input frames."""

        pil_images = tensor_to_pil(frames)

        if not pil_images:
            return {"ui": {"gifs": []}}

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename in output dir
        counter = 0
        while True:
            filename = f"preview_{counter:05d}.webm"
            full_path = os.path.join(output_dir, filename)
            if not os.path.exists(full_path):
                break
            counter += 1

        # Write frames to a temp dir, then encode to webm with ffmpeg
        with tempfile.TemporaryDirectory() as frame_dir:
            for idx, img in enumerate(pil_images):
                img.save(os.path.join(frame_dir, f"frame_{idx:05d}.png"))

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(frame_dir, "frame_%05d.png"),
                # VP9 in a webm container — browser-compatible, supports alpha
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuva420p",
                # Quality-based encoding (crf 30, no bitrate limit)
                "-crf",
                "30",
                "-b:v",
                "0",
                # Loop flag (webm/VP9 containers don't carry a loop count the
                # same way GIF does; looping is controlled by the player)
                full_path,
            ]

            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, timeout=120
            )

        if result.returncode != 0:
            # Fallback: animated GIF
            filename = f"preview_{counter:05d}.gif"
            full_path = os.path.join(output_dir, filename)
            pil_images[0].save(
                full_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=int(1000 / fps),
                loop=0,
                optimize=False,
            )
            preview = {
                "filename": filename,
                "subfolder": "",
                "type": "output",
                "format": "image/gif",
                "frame_rate": fps,
            }
            return {"ui": {"gifs": [preview]}}

        preview = {
            "filename": filename,
            "subfolder": "",
            "type": "output",
            # "video/webm" tells the VHS frontend to use a <video> element
            "format": "video/webm",
            "frame_rate": fps,
        }

        return {"ui": {"gifs": [preview]}}
