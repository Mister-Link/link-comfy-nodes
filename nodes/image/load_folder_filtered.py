from __future__ import annotations

import os
from typing import Any

import folder_paths
from aiohttp import web
from server import PromptServer

from .load_folder import (
    BIGMAX,
    _build_preview_images,
    _is_changed_load_folder,
    _load_images,
    _resolve_load_folder_directory,
    _validate_load_folder,
)


ALL_PREFIX = "all"


def _category_prefix(name: str) -> str:
    # Output folders use a leading marker such as "[O]". Keep that marker in
    # the actual directory value, but omit it from the user-facing category.
    if name.startswith("[") and "]" in name:
        name = name.split("]", 1)[1]
    return name.split("_", 1)[0].casefold()


def _directory_records() -> list[dict[str, Any]]:
    """Return input subdirectories newest first, including their category prefix."""
    input_dir = folder_paths.get_input_directory()
    records = []
    try:
        entries = os.scandir(input_dir)
    except OSError:
        return records

    with entries:
        for entry in entries:
            if entry.name == "clipspace":
                continue
            try:
                if not entry.is_dir():
                    continue
                stat = entry.stat()
            except OSError:
                continue

            prefix = _category_prefix(entry.name)
            records.append(
                {
                    "name": entry.name,
                    "prefix": prefix,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )

    records.sort(key=lambda item: (-item["mtime_ns"], item["name"].casefold()))
    return records


def _matches_prefix(name: str, prefix: str) -> bool:
    prefix = prefix.strip().casefold()
    if not prefix or prefix == ALL_PREFIX:
        return True
    return _category_prefix(name) == prefix


def _directory_payload(prefix: str = ALL_PREFIX) -> dict[str, list[str]]:
    records = _directory_records()
    prefixes = []
    seen = set()
    for record in records:
        if record["prefix"] not in seen:
            prefixes.append(record["prefix"])
            seen.add(record["prefix"])

    directories = [
        record["name"]
        for record in records
        if _matches_prefix(record["name"], prefix)
    ]
    return {"directories": directories, "prefixes": prefixes}


@PromptServer.instance.routes.get("/link_comfy/load_folder_filtered_directories")
async def _load_folder_filtered_directories(request):
    prefix = request.rel_url.query.get("prefix", ALL_PREFIX)
    return web.json_response(_directory_payload(prefix))


class LoadFolderFilteredNode:
    """Load images from a newest-first, prefix-filtered input folder list."""

    CATEGORY = "image/io"
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count")
    FUNCTION = "load_images"

    @classmethod
    def INPUT_TYPES(cls):
        payload = _directory_payload()
        return {
            "required": {
                "directory_prefix": ([ALL_PREFIX, *payload["prefixes"]],),
                "directory": (payload["directories"],),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
        }

    def load_images(self, directory: str, directory_prefix: str = ALL_PREFIX, **kwargs):
        directory = _resolve_load_folder_directory(directory)
        images, masks, frame_count = _load_images(directory, **kwargs)
        preview_images = _build_preview_images(directory, **kwargs)
        return {
            "ui": {"fast_images": preview_images},
            "result": (images, masks, frame_count),
        }

    @classmethod
    def IS_CHANGED(cls, directory: str, directory_prefix: str = ALL_PREFIX, **kwargs):
        directory = _resolve_load_folder_directory(directory)
        return _is_changed_load_folder(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(cls, directory: str, directory_prefix: str = ALL_PREFIX, **kwargs):
        directory = _resolve_load_folder_directory(directory)
        return _validate_load_folder(directory)
