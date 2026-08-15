from __future__ import annotations

import hashlib
import html
import os
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
from aiohttp import web
from comfy.k_diffusion.utils import FolderOfImages
from comfy.utils import ProgressBar
from server import PromptServer

BIGMAX = (2**53 - 1)


def _strip_path(path: str) -> str:
    path = path.strip()
    # Some UIs pass folder names through HTML escaping one or more times.
    # Normalize repeatedly so values like "&amp;apos;folder&amp;apos;" resolve
    # back to a plain folder name before ComfyUI validates the path.
    for _ in range(3):
        unescaped = html.unescape(path)
        if unescaped == path:
            break
        path = unescaped

    if len(path) >= 2 and path[0] == path[-1] and path[0] in {"'", '"'}:
        path = path[1:-1]
    else:
        if path.startswith(("'", '"')):
            path = path[1:]
        if path.endswith(("'", '"')):
            path = path[:-1]
    return path


def _resolve_load_folder_directory(name: str) -> str:
    name = _strip_path(name)
    input_dir = folder_paths.get_input_directory()
    candidate = os.path.normpath(os.path.join(input_dir, name))
    # LoadFolderNode's directory choices are single-level entries taken
    # straight from os.listdir(input_dir) (see INPUT_TYPES), including
    # symlinks that batch_save.py plants there to expose output folders.
    # Those symlinks legitimately resolve outside input_dir, so we can't use
    # folder_paths.get_annotated_filepath's realpath-based containment check
    # here -- it follows the symlink target and rejects it as an escape.
    # Instead just make sure `name` didn't smuggle a "/" or ".." to escape
    # input_dir before symlink resolution.
    if os.path.dirname(candidate) != os.path.normpath(input_dir):
        raise ValueError("Invalid file path: {!r}".format(name))
    return candidate


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
    # Palette-mode ('P') PNGs -- the common export format for pixel art / sprite
    # sheets -- carry transparency via a tRNS table that doesn't show up in
    # getbands() until after converting to RGBA, so checking bands alone
    # silently drops real per-pixel transparency on indexed images.
    has_alpha = 'A' in i.getbands() or 'transparency' in i.info
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


def _build_preview_images(
    directory: str,
    image_load_cap: int = 0,
    skip_first_images: int = 0,
    select_every_nth: int = 1,
):
    input_dir = folder_paths.get_input_directory()
    dir_files = _get_sorted_dir_files(
        directory,
        skip_first_images,
        select_every_nth,
        FolderOfImages.IMG_EXTENSIONS,
    )
    if image_load_cap > 0:
        dir_files = dir_files[:image_load_cap]

    preview_images = []
    for file_path in dir_files:
        relative_path = os.path.relpath(file_path, input_dir)
        subfolder = os.path.dirname(relative_path)
        filename = os.path.basename(relative_path)

        with Image.open(file_path) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size

        preview_images.append(
            {
                "filename": filename,
                "full_filename": filename,
                "subfolder": "" if subfolder == "." else subfolder,
                "type": "input",
                "width": width,
                "height": height,
            }
        )

    return preview_images


def _query_int(query, name: str, default: int) -> int:
    try:
        return int(query.get(name, default))
    except (TypeError, ValueError):
        return default


# Lets the frontend fetch the frame list for the animated preview as soon as
# the node loads or the directory/paging widgets change, instead of only
# after the workflow actually runs.
@PromptServer.instance.routes.get("/link_comfy/load_folder_preview")
async def _load_folder_preview_route(request):
    directory = request.rel_url.query.get("directory", "")
    image_load_cap = _query_int(request.rel_url.query, "image_load_cap", 0)
    skip_first_images = _query_int(request.rel_url.query, "skip_first_images", 0)
    select_every_nth = _query_int(request.rel_url.query, "select_every_nth", 1) or 1

    try:
        resolved = _resolve_load_folder_directory(directory)
    except ValueError:
        return web.json_response({"frames": []})

    if not os.path.isdir(resolved):
        return web.json_response({"frames": []})

    frames = _build_preview_images(
        resolved,
        image_load_cap=image_load_cap,
        skip_first_images=skip_first_images,
        select_every_nth=select_every_nth,
    )
    return web.json_response({"frames": frames})


class LoadFolderNode:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        directories = [
            item for item in os.listdir(input_dir)
            if not os.path.isfile(os.path.join(input_dir, item)) and item != "clipspace"
        ]
        directories.sort(key=str.lower)
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
        directory = _resolve_load_folder_directory(directory)
        result = _load_images(directory, **kwargs)
        preview_images = _build_preview_images(directory, **kwargs)
        return {"ui": {"fast_images": preview_images}, "result": result}

    @classmethod
    def IS_CHANGED(cls, directory: str, **kwargs):
        directory = _resolve_load_folder_directory(directory)
        return _is_changed_load_folder(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(cls, directory: str, **kwargs):
        directory = _resolve_load_folder_directory(directory)
        return _validate_load_folder(directory)
