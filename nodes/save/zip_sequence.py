from __future__ import annotations

import json
import os
import zipfile
from io import BytesIO

import numpy as np
import torch
from PIL import Image

import folder_paths  # type: ignore[import-untyped]


class SaveImageSequenceZip:
    CATEGORY: str = "Video/Masking"
    RETURN_TYPES: tuple[str, ...] = ()
    FUNCTION: str = "save_sequence"
    OUTPUT_NODE: bool = True
    INPUT_IS_LIST: bool = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zip_path": (
                    "STRING",
                    {"default": "output/archive", "multiline": False},
                ),
            },
            "optional": {
                "input1_prefix": ("STRING", {"default": ""}),
                "input2_prefix": ("STRING", {"default": ""}),
                "input3_prefix": ("STRING", {"default": ""}),
                "input4_prefix": ("STRING", {"default": ""}),
                "input5_prefix": ("STRING", {"default": ""}),
                "input6_prefix": ("STRING", {"default": ""}),
                "input7_prefix": ("STRING", {"default": ""}),
                "input8_prefix": ("STRING", {"default": ""}),
            },
        }

    def save_sequence(self, zip_path: str, **kwargs: object):
        if isinstance(zip_path, (list, tuple)):
            zip_path = zip_path[0] if zip_path else ""

        inputs = []
        for index in range(1, 9):
            data = kwargs.get(f"input{index}")
            if data is None:
                continue
            prefix_value = kwargs.get(f"input{index}_prefix", f"input{index}")
            if isinstance(prefix_value, (list, tuple)):
                prefix_value = prefix_value[0] if prefix_value else ""
            prefix_text = prefix_value if isinstance(prefix_value, str) else ""
            prefix_text = prefix_text.strip() or f"input{index}"
            inputs.append((f"input{index}", data, prefix_text))

        zip_path = zip_path.strip()

        def format_path(template: str, idx: int) -> str:
            if "{index" in template:
                return template.format_map({"index": idx})
            return template.format(idx)

        output_dir = folder_paths.get_output_directory()

        uses_template = False
        if "{" in zip_path and "}" in zip_path:
            try:
                test_candidate = format_path(zip_path, 1)
                uses_template = test_candidate != zip_path
            except (ValueError, KeyError, IndexError):
                uses_template = False

        if uses_template:
            index = 1
            while True:
                try:
                    candidate = format_path(zip_path, index)
                except (ValueError, KeyError, IndexError):
                    uses_template = False
                    break
                if not candidate.endswith(".zip"):
                    candidate = f"{candidate}.zip"
                full_candidate = (
                    candidate
                    if os.path.isabs(candidate)
                    else os.path.join(output_dir, candidate)
                )
                if not os.path.exists(full_candidate):
                    zip_path = candidate
                    full_zip_path = full_candidate
                    break
                index += 1

        if not uses_template:
            if not zip_path.endswith(".zip"):
                zip_path = f"{zip_path}.zip"
            if os.path.isabs(zip_path):
                full_zip_path = zip_path
            else:
                full_zip_path = os.path.join(output_dir, zip_path)

        zip_dir = os.path.dirname(full_zip_path)
        if zip_dir:
            os.makedirs(zip_dir, exist_ok=True)

        used_names: set[str] = set()

        def reserve_name(file_name: str) -> str:
            if file_name not in used_names:
                used_names.add(file_name)
                return file_name
            base, ext = os.path.splitext(file_name)
            counter = 1
            while True:
                candidate = f"{base}_{counter:02d}{ext}"
                if candidate not in used_names:
                    used_names.add(candidate)
                    return candidate
                counter += 1

        with zipfile.ZipFile(full_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for _input_name, data, prefix in inputs:
                if data is None:
                    continue
                if isinstance(data, str):
                    payload = data.encode("utf-8")
                    ext = self._extension_for_text(data)
                    file_name = reserve_name(f"{prefix}.{ext}")
                    zipf.writestr(file_name, payload)
                    continue
                if isinstance(data, dict):
                    payload = json.dumps(data, indent=2).encode("utf-8")
                    file_name = reserve_name(f"{prefix}.json")
                    zipf.writestr(file_name, payload)
                    continue
                if isinstance(data, (list, tuple)) and data:
                    if all(isinstance(item, str) for item in data):
                        for i, item in enumerate(data, start=1):
                            payload = item.encode("utf-8")
                            ext = self._extension_for_text(item)
                            file_name = reserve_name(f"{prefix}.{ext}")
                            zipf.writestr(file_name, payload)
                        continue
                    if all(isinstance(item, dict) for item in data):
                        for i, item in enumerate(data, start=1):
                            payload = json.dumps(item, indent=2).encode("utf-8")
                            file_name = reserve_name(f"{prefix}.json")
                            zipf.writestr(file_name, payload)
                        continue
                frames = self._normalize_frames(data)

                prefix_base, prefix_ext = os.path.splitext(prefix)
                has_extension = bool(prefix_ext)

                if has_extension:
                    ext = prefix_ext.lstrip(".")
                    format_name = (
                        ext.upper()
                        if ext.upper() in {"PNG", "JPEG", "JPG", "WEBP"}
                        else "PNG"
                    )
                    if format_name == "JPG":
                        format_name = "JPEG"
                else:
                    ext = "png"
                    format_name = "PNG"

                if len(frames) == 1:
                    pil_img = self._to_pil(frames[0], ext)
                    img_buffer = BytesIO()
                    pil_img.save(img_buffer, format=format_name)
                    img_buffer.seek(0)

                    image_filename = reserve_name(
                        f"{prefix_base}{prefix_ext}"
                        if has_extension
                        else f"{prefix_base}.{ext}"
                    )

                    zipf.writestr(image_filename, img_buffer.getvalue())
                else:
                    for i in range(len(frames)):
                        frame_index = i + 1
                        pil_img = self._to_pil(frames[i], ext)
                        img_buffer = BytesIO()
                        pil_img.save(img_buffer, format=format_name)
                        img_buffer.seek(0)

                        if "{" in prefix_base:
                            try:
                                image_filename = prefix_base.format(frame_index) + (
                                    prefix_ext if has_extension else f".{ext}"
                                )
                            except (KeyError, IndexError):
                                digits = max(2, len(str(len(frames))))
                                image_filename = (
                                    f"{prefix}_{frame_index:0{digits}d}.{ext}"
                                    if not has_extension
                                    else f"{prefix_base}_{frame_index:0{digits}d}{prefix_ext}"
                                )
                        else:
                            digits = max(2, len(str(len(frames))))
                            image_filename = (
                                f"{prefix}_{frame_index:0{digits}d}.{ext}"
                                if not has_extension
                                else f"{prefix_base}_{frame_index:0{digits}d}{prefix_ext}"
                            )

                        image_filename = reserve_name(image_filename)
                        zipf.writestr(image_filename, img_buffer.getvalue())

        zip_filename = os.path.basename(zip_path)

        zip_path_for_url = zip_path.strip("/")
        if zip_path_for_url.startswith("output/"):
            zip_path_for_url = zip_path_for_url[len("output/"):]
        elif zip_path_for_url.startswith("output"):
            zip_path_for_url = zip_path_for_url[len("output"):].lstrip("/")

        download_url = f"/view?filename={zip_path_for_url}&type=output"

        return {
            "ui": {
                "text": [
                    f'<a href="{download_url}" target="_blank" style="color: #4a9eff; text-decoration: underline;">Download: {zip_filename}</a>'
                ],
            }
        }

    @staticmethod
    def _extension_for_text(text: str) -> str:
        trimmed = text.lstrip()
        if trimmed.startswith("{") or trimmed.startswith("["):
            return "json"
        return "txt"

    @staticmethod
    def _normalize_frames(data: torch.Tensor | np.ndarray | list | tuple) -> list[np.ndarray]:
        # Returns a flat list of (H, W, C) frames rather than a single stacked
        # array, since frames may not all share the same dimensions.
        if isinstance(data, (list, tuple)):
            frames_list: list[np.ndarray] = []
            for item in data:
                frames_list.extend(SaveImageSequenceZip._normalize_frames(item))
            if not frames_list:
                raise ValueError("Expected non-empty list/tuple for frames")
            return frames_list
        if isinstance(data, np.ndarray):
            frames = torch.from_numpy(data).float()
        else:
            frames = data.detach().cpu().float()
        if frames.ndim == 2:
            frames = frames.unsqueeze(0).unsqueeze(-1)
        elif frames.ndim == 3:
            if frames.shape[-1] in {1, 3, 4}:
                frames = frames.unsqueeze(0)
            else:
                frames = frames.unsqueeze(-1)
        elif frames.ndim == 4 and frames.shape[-1] not in {1, 3, 4}:
            frames = frames.unsqueeze(-1)
        if frames.ndim != 4:
            raise ValueError("Expected data with shape (N, H, W, C)")
        return [frames[i].numpy() for i in range(frames.shape[0])]

    @staticmethod
    def _to_pil(frame: np.ndarray, ext: str) -> Image.Image:
        frame = np.clip(frame, 0.0, 1.0)
        if frame.shape[-1] == 1:
            frame_255 = (frame[..., 0] * 255).astype(np.uint8)
            return Image.fromarray(frame_255, mode="L")
        if frame.shape[-1] == 4 and ext in {"jpg", "jpeg"}:
            frame = frame[..., :3]
        frame_255 = (frame * 255).astype(np.uint8)
        mode = "RGBA" if frame_255.shape[-1] == 4 else "RGB"
        return Image.fromarray(frame_255, mode=mode)
