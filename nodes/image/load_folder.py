from __future__ import annotations

import hashlib
import os
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
from comfy.k_diffusion.utils import FolderOfImages
from comfy.utils import ProgressBar

BIGMAX = (2**53 - 1)


def _strip_path(path: str) -> str:
    path = path.strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path


def _calculate_file_hash(filename: str) -> str:
    # modified time is used instead of file contents for speed on large files
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()


def _get_sorted_dir_files(directory: str, skip_first_images: int = 0, select_every_nth: int = 1, extensions: Iterable = None):
    directory = _strip_path(directory)
    dir_files = sorted(os.listdir(directory))
    dir_files = [os.path.join(directory, x) for x in dir_files]
    dir_files = list(filter(os.path.isfile, dir_files))
    if extensions is not None:
        extensions = list(extensions)
        dir_files = [f for f in dir_files if "." + f.split(".")[-1].lower() in extensions]
    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    return dir_files


def _is_changed_load_folder(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1, **kwargs):
    if not os.path.isdir(directory):
        return False
    dir_files = _get_sorted_dir_files(directory, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)
    if image_load_cap != 0:
        dir_files = dir_files[:image_load_cap]
    m = hashlib.sha256()
    for filepath in dir_files:
        m.update(_calculate_file_hash(filepath).encode())
    return m.digest().hex()


def _validate_load_folder(directory: str):
    if not os.path.isdir(directory):
        return f"Directory '{directory}' cannot be found."
    if len(os.listdir(directory)) == 0:
        return f"No files in directory '{directory}'."
    return True


def _load_image(file_path: str):
    i = Image.open(file_path)
    # exif_transpose can only ever rotate, but rotating can swap width/height
    i = ImageOps.exif_transpose(i)
    has_alpha = 'A' in i.getbands()
    i = i.convert("RGBA" if has_alpha else "RGB")
    arr = np.array(i, dtype=np.float32) / 255.0
    if has_alpha:
        mask = 1 - arr[:, :, 3]
        arr = arr[:, :, :3]
    else:
        mask = np.zeros(arr.shape[:2], dtype=np.float32)
    image = torch.from_numpy(arr).unsqueeze(0)
    mask = torch.from_numpy(mask).unsqueeze(0)
    return image, mask


def _load_images(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
    dir_files = _get_sorted_dir_files(directory, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)
    if len(dir_files) == 0:
        raise FileNotFoundError(f"No files in directory '{directory}'.")
    if image_load_cap > 0:
        dir_files = dir_files[:image_load_cap]

    total_images = len(dir_files)
    pbar = ProgressBar(total_images)
    images = []
    masks = []
    first_shape = None
    for idx, file_path in enumerate(dir_files):
        image, mask = _load_image(file_path)
        if first_shape is None:
            first_shape = image.shape
        elif image.shape != first_shape:
            raise ValueError(
                "Image size mismatch. All images in the folder must have the same dimensions."
            )
        images.append(image)
        masks.append(mask)
        pbar.update_absolute(idx + 1, total_images)

    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")
    return torch.cat(images, dim=0), torch.cat(masks, dim=0), len(images)


class LoadFolderNode:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        directories = [
            item for item in os.listdir(input_dir)
            if not os.path.isfile(os.path.join(input_dir, item)) and item != "clipspace"
        ]
        return {
            "required": {
                "directory": (directories,),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count")
    FUNCTION = "load_images"
    CATEGORY = "image/io"

    def load_images(self, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(_strip_path(directory))
        return _load_images(directory, **kwargs)

    @classmethod
    def IS_CHANGED(cls, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(_strip_path(directory))
        return _is_changed_load_folder(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(cls, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(_strip_path(directory))
        return _validate_load_folder(directory)
