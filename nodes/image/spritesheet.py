from __future__ import annotations

import json
import math

import numpy as np
import torch


class SpritesheetBuilderNode:
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("spritesheet", "frames", "metadata")
    FUNCTION = "build_spritesheet"
    CATEGORY = "image/transform"
    INPUT_IS_LIST = True

    _ASPECT_RATIOS = {
        "1:1 (Square)": (1, 1),
        "4:3 (Landscape)": (4, 3),
        "3:4 (Portrait)": (3, 4),
        "16:9 (Landscape)": (16, 9),
        "9:16 (Portrait)": (9, 16),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "aspect_ratio": (list(cls._ASPECT_RATIOS.keys()),),
            },
            "optional": {
                "stabilization_metadata": ("STRING", {"forceInput": True}),
            },
        }

    @staticmethod
    def _flatten_frames(value) -> list[torch.Tensor]:
        if isinstance(value, (list, tuple)):
            frames: list[torch.Tensor] = []
            for item in value:
                frames.extend(SpritesheetBuilderNode._flatten_frames(item))
            return frames
        tensor = value.detach().cpu().float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError("Expected frames with shape (N, H, W, C)")
        return [tensor[i] for i in range(tensor.shape[0])]

    @staticmethod
    def _one_value(value, default=""):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value if value is not None else default

    @classmethod
    def _parse_stabilization_metadata(cls, value, frame_count: int):
        raw = cls._one_value(value, "")
        if not raw:
            return None
        try:
            manifest = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("stabilization_metadata must be valid JSON") from exc
        if manifest.get("format") != "link-comfy-nodes/stabilization-v1":
            raise ValueError("Unsupported stabilization metadata format")
        records = manifest.get("frames")
        if not isinstance(records, list) or len(records) != frame_count:
            raise ValueError("Stabilization metadata frame count must match the image batch")
        source_size = manifest.get("sourceSize", {})
        pivot = manifest.get("pivot", {})
        if int(source_size.get("w", 0)) < 1 or int(source_size.get("h", 0)) < 1:
            raise ValueError("Stabilization metadata is missing sourceSize")
        if "x" not in pivot or "y" not in pivot:
            raise ValueError("Stabilization metadata is missing pivot")
        return manifest

    def build_spritesheet(self, frames, aspect_ratio="1:1 (Square)", stabilization_metadata=""):
        aspect_ratio = self._one_value(aspect_ratio, "1:1 (Square)")
        frame_list = self._flatten_frames(frames)
        if not frame_list:
            raise ValueError("Expected at least one frame")
        frame_count = len(frame_list)
        manifest = self._parse_stabilization_metadata(stabilization_metadata, frame_count)

        placement = None
        if manifest is not None:
            placement = manifest["frames"]
            trimmed = []
            for index, frame in enumerate(frame_list):
                # The image arriving through ComfyUI is authoritative for the
                # crop. It can differ by a pixel from metadata generated before
                # graph serialization or a node reload, so never reject a valid
                # alpha-bearing frame solely because its saved rectangle is
                # marginally outside the current canvas.
                frame_np = frame.numpy()
                if frame_np.shape[-1] == 4:
                    ys, xs = np.where(frame_np[..., 3] > 0.02)
                else:
                    ys, xs = np.array([]), np.array([])

                if xs.size:
                    x, y = int(xs.min()), int(ys.min())
                    w, h = int(xs.max()) + 1 - x, int(ys.max()) + 1 - y
                else:
                    rect = placement[index].get("spriteSourceSize", {})
                    x, y = int(rect.get("x", 0)), int(rect.get("y", 0))
                    x = max(0, min(x, frame.shape[1] - 1))
                    y = max(0, min(y, frame.shape[0] - 1))
                    w = max(1, min(int(rect.get("w", frame.shape[1])), frame.shape[1] - x))
                    h = max(1, min(int(rect.get("h", frame.shape[0])), frame.shape[0] - y))

                placement[index]["spriteSourceSize"] = {"x": x, "y": y, "w": w, "h": h}
                trimmed.append(frame[y:y + h, x:x + w])
            frame_list = trimmed

        target_ratio = self._aspect_ratio_value(aspect_ratio)
        frame_width = max(frame.shape[1] for frame in frame_list)
        frame_height = max(frame.shape[0] for frame in frame_list)
        use_alpha = any(frame.shape[-1] == 4 for frame in frame_list)
        frame_channels = 4 if use_alpha else 3

        columns, rows = self._closest_grid(frame_count, target_ratio, frame_width, frame_height)
        sheet_width = frame_width * columns
        sheet_height = frame_height * rows
        spritesheet = np.zeros((sheet_height, sheet_width, frame_channels), dtype=np.float32)
        frames_out = np.zeros((frame_count, frame_height, frame_width, frame_channels), dtype=np.float32)
        frame_metadata = []

        for index, frame in enumerate(frame_list):
            frame_np = frame.numpy()
            h, w = frame_np.shape[:2]
            row, col = divmod(index, columns)
            y0, x0 = row * frame_height, col * frame_width
            spritesheet[y0:y0 + h, x0:x0 + w, :3] = frame_np[:, :, :3]
            frames_out[index, :h, :w, :3] = frame_np[:, :, :3]
            if frame_channels == 4:
                alpha = frame_np[:, :, 3] if frame_np.shape[-1] == 4 else 1.0
                spritesheet[y0:y0 + h, x0:x0 + w, 3] = alpha
                frames_out[index, :h, :w, 3] = alpha

            record = {"index": index, "frame": {"x": x0, "y": y0, "w": w, "h": h}}
            if manifest is not None:
                record["sourceSize"] = manifest["sourceSize"]
                record["spriteSourceSize"] = placement[index]["spriteSourceSize"]
                record["pivot"] = manifest["pivot"]
            frame_metadata.append(record)

        result = torch.from_numpy(spritesheet).unsqueeze(0)
        frames_tensor = torch.from_numpy(frames_out)
        metadata = {
            "spritesheet": {
                "width": sheet_width,
                "height": sheet_height,
                "rows": rows,
                "columns": columns,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "frame_count": frame_count,
            },
            "frames": frame_metadata,
        }
        if manifest is not None:
            metadata["stabilization"] = {
                "sourceSize": manifest["sourceSize"],
                "pivot": manifest["pivot"],
            }
        return (result, frames_tensor, json.dumps(metadata, indent=2))

    @classmethod
    def _aspect_ratio_value(cls, aspect_ratio: str) -> float:
        width, height = cls._ASPECT_RATIOS[aspect_ratio]
        return width / height

    @staticmethod
    def _closest_grid(frame_count: int, target_ratio: float, frame_width: int, frame_height: int) -> tuple[int, int]:
        best_cols, best_rows, best_diff = 1, frame_count, float("inf")
        for cols in range(1, frame_count + 1):
            rows = math.ceil(frame_count / cols)
            diff = abs((cols * frame_width) / (rows * frame_height) - target_ratio)
            if diff < best_diff:
                best_cols, best_rows, best_diff = cols, rows, diff
        return best_cols, best_rows
