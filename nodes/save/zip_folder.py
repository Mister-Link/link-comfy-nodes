from __future__ import annotations

import os
import zipfile

import folder_paths  # type: ignore[import-untyped]


class SaveFolderAsZip:
    CATEGORY = "utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zip_path",)
    FUNCTION = "save_folder_as_zip"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
            },
        }

    def save_folder_as_zip(self, folder_path: str):
        folder_path = folder_path.strip()

        if not folder_path:
            raise ValueError("Folder path cannot be empty")

        output_dir = folder_paths.get_output_directory()

        if os.path.isabs(folder_path):
            full_folder_path = folder_path
        else:
            full_folder_path = os.path.join(output_dir, folder_path)

        if not os.path.exists(full_folder_path):
            raise ValueError(f"Folder does not exist: {full_folder_path}")

        if not os.path.isdir(full_folder_path):
            raise ValueError(f"Path is not a directory: {full_folder_path}")

        full_zip_path = f"{full_folder_path}.zip"

        if os.path.exists(full_zip_path):
            os.remove(full_zip_path)

        with zipfile.ZipFile(full_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(full_folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(
                        file_path, os.path.dirname(full_folder_path)
                    )
                    zipf.write(file_path, arcname)

        print(f"Created zip file: {full_zip_path}")

        try:
            zip_path_relative = os.path.relpath(full_zip_path, output_dir)
        except ValueError:
            zip_path_relative = full_zip_path

        return (zip_path_relative,)
