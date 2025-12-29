"""String manipulation nodes."""

import os
import shutil
import zipfile

import folder_paths


class AdvancedStringConcat:
    """Concatenate strings using template placeholders like %1, %2, etc."""

    CATEGORY = "utils"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "concat_strings"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": (
                    "STRING",
                    {"default": "%1 %2", "multiline": True},
                ),
            },
            "optional": {
                "string1": ("STRING", {"default": "", "forceInput": True}),
                "string2": ("STRING", {"default": "", "forceInput": True}),
                "string3": ("STRING", {"default": "", "forceInput": True}),
                "string4": ("STRING", {"default": "", "forceInput": True}),
                "string5": ("STRING", {"default": "", "forceInput": True}),
                "string6": ("STRING", {"default": "", "forceInput": True}),
                "string7": ("STRING", {"default": "", "forceInput": True}),
                "string8": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    def concat_strings(self, template: str, **kwargs):
        """Replace %1, %2, etc. in template with corresponding string inputs."""
        result = template

        # Replace placeholders %1 through %8
        for i in range(1, 9):
            string_key = f"string{i}"
            string_value = kwargs.get(string_key, "")
            if string_value:
                result = result.replace(f"%{i}", string_value)

        return (result,)


class SaveFolderAsZip:
    """Convert an existing folder to a zip file."""

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
        """Create a zip file from the specified folder."""
        folder_path = folder_path.strip()

        if not folder_path:
            raise ValueError("Folder path cannot be empty")

        # Get output directory
        output_dir = folder_paths.get_output_directory()

        # If folder_path is relative, resolve it relative to output directory
        if os.path.isabs(folder_path):
            full_folder_path = folder_path
        else:
            full_folder_path = os.path.join(output_dir, folder_path)

        if not os.path.exists(full_folder_path):
            raise ValueError(f"Folder does not exist: {full_folder_path}")

        if not os.path.isdir(full_folder_path):
            raise ValueError(f"Path is not a directory: {full_folder_path}")

        # Create zip file path (same location as folder, with .zip extension)
        full_zip_path = f"{full_folder_path}.zip"

        # Remove existing zip if it exists
        if os.path.exists(full_zip_path):
            os.remove(full_zip_path)

        # Create the zip file
        with zipfile.ZipFile(full_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the folder and add all files
            for root, dirs, files in os.walk(full_folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Calculate the archive name (relative path from the folder)
                    arcname = os.path.relpath(
                        file_path, os.path.dirname(full_folder_path)
                    )
                    zipf.write(file_path, arcname)

        print(f"Created zip file: {full_zip_path}")

        # Return path relative to output folder
        try:
            zip_path_relative = os.path.relpath(full_zip_path, output_dir)
        except ValueError:
            # If can't make it relative (e.g., different drives on Windows), return absolute
            zip_path_relative = full_zip_path

        return (zip_path_relative,)


class PreviewAsMarkdown:
    """Preview a string as rendered markdown on the node."""

    CATEGORY = "utils"
    RETURN_TYPES = ()
    FUNCTION = "preview_markdown"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
            },
        }

    def preview_markdown(self, source: str):
        """Pass through the string and send it to the UI for markdown rendering."""
        return {
            "ui": {"markdown": [source]},
        }


NODE_CLASS_MAPPINGS = {
    "AdvancedStringConcat": AdvancedStringConcat,
    "SaveFolderAsZip": SaveFolderAsZip,
    "PreviewAsMarkdown": PreviewAsMarkdown,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedStringConcat": "Concat",
    "SaveFolderAsZip": "Save Folder as ZIP",
    "PreviewAsMarkdown": "Preview as Markdown",
}
