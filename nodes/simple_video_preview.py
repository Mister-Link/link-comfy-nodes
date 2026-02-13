"""Simple Video Preview Node - A streamlined video preview node with minimal inputs."""

import os
import tempfile
from typing import Any

import numpy as np
import torch
from PIL import Image

import folder_paths  # type: ignore[import-untyped]


def tensor_to_pil(tensor: torch.Tensor) -> list[Image.Image]:
    """Convert a tensor batch to PIL images."""
    # Handle different tensor shapes
    if tensor.ndim == 3:
        # Single image [H, W, C]
        tensor = tensor.unsqueeze(0)

    # Convert from [B, H, W, C] format
    images = []
    for img_tensor in tensor:
        # Ensure values are in 0-255 range
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

        # Convert tensor to PIL images
        pil_images = tensor_to_pil(frames)

        if not pil_images:
            return {"ui": {"gifs": []}}

        # Create workflow assets output directory
        assets_subfolder = "workflow_assets"
        output_dir = os.path.join(folder_paths.get_output_directory(), assets_subfolder)
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename
        # Use a counter-based approach for cleaner filenames
        counter = 0
        while True:
            filename = f"preview_{counter:05d}.webm"
            full_path = os.path.join(output_dir, filename)
            if not os.path.exists(full_path):
                break
            counter += 1

        # Save as WebM using ffmpeg
        try:
            import subprocess

            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save frames as temporary images
                for idx, img in enumerate(pil_images):
                    frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
                    img.save(frame_path)

                # Use ffmpeg to create WebM
                # -y: overwrite output file
                # -framerate: input framerate
                # -i: input pattern
                # -c:v: video codec (libvpx-vp9 for WebM)
                # -pix_fmt: pixel format
                # -lossless 1: lossless encoding for quality
                # -loop 0: loop infinitely
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    os.path.join(temp_dir, "frame_%05d.png"),
                    "-c:v",
                    "libvpx-vp9",
                    "-pix_fmt",
                    "yuva420p",
                    "-lossless",
                    "1",
                    "-loop",
                    "0",
                    full_path,
                ]

                result = subprocess.run(
                    ffmpeg_cmd, capture_output=True, text=True, timeout=60
                )

                if result.returncode != 0:
                    # Fallback to GIF if WebM fails
                    filename = filename.replace(".webm", ".gif")
                    full_path = full_path.replace(".webm", ".gif")

                    # Save as GIF
                    pil_images[0].save(
                        full_path,
                        save_all=True,
                        append_images=pil_images[1:],
                        duration=int(1000 / fps),  # duration in milliseconds
                        loop=0,
                        optimize=False,
                    )

        except (ImportError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            # Fallback to GIF if ffmpeg is not available or fails
            filename = filename.replace(".webm", ".gif")
            full_path = full_path.replace(".webm", ".gif")

            # Save as GIF with looping
            pil_images[0].save(
                full_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=int(1000 / fps),  # duration in milliseconds
                loop=0,
                optimize=False,
            )

        # Determine format based on actual saved file
        format_type = "image/webm" if full_path.endswith(".webm") else "image/gif"

        # Return preview data in the format expected by ComfyUI
        # This enables the context menu options
        preview = {
            "filename": filename,
            "subfolder": assets_subfolder,
            "type": "output",
            "format": format_type,
        }

        # The "ui" key with "gifs" enables preview display and context menu
        return {"ui": {"gifs": [preview]}}
