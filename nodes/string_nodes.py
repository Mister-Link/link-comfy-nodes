"""String manipulation nodes."""

import os
import shutil
import zipfile


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

        if not os.path.exists(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        # Create zip file path (same location as folder, with .zip extension)
        zip_path = f"{folder_path}.zip"

        # Remove existing zip if it exists
        if os.path.exists(zip_path):
            os.remove(zip_path)

        # Create the zip file
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the folder and add all files
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Calculate the archive name (relative path from the folder)
                    arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                    zipf.write(file_path, arcname)

        print(f"Created zip file: {zip_path}")

        return (zip_path,)


NODE_CLASS_MAPPINGS = {
    "AdvancedStringConcat": AdvancedStringConcat,
    "SaveFolderAsZip": SaveFolderAsZip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedStringConcat": "Advanced String Concat",
    "SaveFolderAsZip": "Save Folder as ZIP",
}
