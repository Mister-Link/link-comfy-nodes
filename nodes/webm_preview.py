from __future__ import annotations

import os
import tempfile
import uuid

import imageio.v3 as iio
import numpy as np
import torch

try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None  # type: ignore

try:
    from aiohttp import web

    from server import PromptServer  # type: ignore

    @PromptServer.instance.routes.get("/webm_preview/stream")
    async def stream_webm(request):
        filename = request.rel_url.query.get("filename", "")
        if not filename or "/" in filename or "\\" in filename or ".." in filename:
            return web.Response(status=400)

        if folder_paths:
            temp_dir = folder_paths.get_temp_directory()
        else:
            temp_dir = tempfile.gettempdir()

        filepath = os.path.join(temp_dir, filename)
        if not os.path.isfile(filepath):
            return web.Response(status=404)

        file_size = os.path.getsize(filepath)
        range_header = request.headers.get("Range")

        with open(filepath, "rb") as f:
            if range_header:
                # Parse "bytes=start-end"
                range_val = range_header.strip().replace("bytes=", "")
                parts = range_val.split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1
                f.seek(start)
                data = f.read(length)
                return web.Response(
                    status=206,
                    body=data,
                    headers={
                        "Content-Type": "video/webm",
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Content-Length": str(length),
                        "Accept-Ranges": "bytes",
                    },
                )
            else:
                data = f.read()
                return web.Response(
                    status=200,
                    body=data,
                    headers={
                        "Content-Type": "video/webm",
                        "Content-Length": str(file_size),
                        "Accept-Ranges": "bytes",
                    },
                )

except Exception:
    pass


class PreviewWebmNode:
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_webm"
    CATEGORY = "image/preview"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.1, "max": 120.0, "step": 0.1},
                ),
            },
        }

    def preview_webm(self, frames: torch.Tensor, fps: float):
        if not isinstance(frames, torch.Tensor) or frames.shape[0] == 0:
            return {"ui": {"webm_preview": []}}

        if folder_paths:
            temp_dir = folder_paths.get_temp_directory()
        else:
            temp_dir = tempfile.gettempdir()

        unique_id = uuid.uuid4().hex[:8]
        filename = f"webm_preview_{unique_id}.webm"
        filepath = os.path.join(temp_dir, filename)

        # frames shape: (N, H, W, C) float32 0-1 RGB
        frame_list = [(f.cpu().numpy() * 255).astype(np.uint8) for f in frames]

        iio.imwrite(
            filepath,
            frame_list,
            fps=fps,
            codec="libvpx",
            pixelformat="yuv420p",
        )

        return {
            "ui": {
                "webm_preview": [
                    {
                        "filename": filename,
                        "subfolder": "",
                        "type": "temp",
                        "frame_count": frames.shape[0],
                        "fps": fps,
                    }
                ]
            }
        }
