from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from .nodes import (
    AdvancedStringConcat,
    AutoCropperNode,
    BatchImageSave,
    BulkBackgroundRemoverBgEraserNode,
    ColorParserNode,
    CropToContentNode,
    FarthestColorNode,
    ImageRotatorNode,
    PixelationDimensionsNode,
    PoseImageSetupNode,
    PreviewAsMarkdown,
    PreviewImageAlpha,
    ReplaceAlpha,
    ResizeImageAndMaskBySideNode,
    SaveFolderAsZip,
    SaveImageSequenceZip,
    SpritesheetBuilderNode,
    VideoMaskEditor,
    WANFrameCalculatorNode,
)
from .nodes.pixel_art.node import ConvertToPixelArt

NODE_CLASS_MAPPINGS = {
    "Hex or 24 Bit Color": ColorParserNode,
    "Farthest Color": FarthestColorNode,
    "WAN Frame Calculator": WANFrameCalculatorNode,
    "Image Rotator": ImageRotatorNode,
    "Remove Background": BulkBackgroundRemoverBgEraserNode,
    "Crop to Content": CropToContentNode,
    "Pixelation Dimensions": PixelationDimensionsNode,
    "Pose Image Setup": PoseImageSetupNode,
    "Resize Image and Mask by Side": ResizeImageAndMaskBySideNode,
    "Spritesheet Builder": SpritesheetBuilderNode,
    "VideoMaskEditor": VideoMaskEditor,
    "PreviewImageAlpha": PreviewImageAlpha,
    "ReplaceAlpha": ReplaceAlpha,
    "Save To Zip": SaveImageSequenceZip,
    "ConvertToPixelArt": ConvertToPixelArt,
    "BatchImageSave": BatchImageSave,
    "Concat": AdvancedStringConcat,
    "Save Folder as ZIP": SaveFolderAsZip,
    "PreviewAsMarkdown": PreviewAsMarkdown,
    "Auto Cropper": AutoCropperNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hex or 24 Bit Color": "Convert Color Format",
    "Farthest Color": "Find Furthest Color",
    "WAN Frame Calculator": "WAN Frame Calculator",
    "Image Rotator": "Rotate Image",
    "Remove Background": "Remove Background",
    "Crop to Content": "Crop to Content",
    "Pixelation Dimensions": "Pixelation Dimensions",
    "Pose Image Setup": "Pose Image Setup",
    "Resize Image and Mask by Side": "Resize Image and Mask by Side",
    "Spritesheet Builder": "Spritesheet Builder",
    "VideoMaskEditor": "Video Mask Editor",
    "PreviewImageAlpha": "Preview Image (Alpha)",
    "ReplaceAlpha": "Replace Alpha",
    "Save To Zip": "Save to ZIP",
    "ConvertToPixelArt": "Convert to Pixel Art",
    "BatchImageSave": "Batch Image Save",
    "Concat": "Concat",
    "Save Folder as ZIP": "Save Folder as ZIP",
    "PreviewAsMarkdown": "Preview as Markdown",
    "Auto Cropper": "Auto Cropper",
}

WEB_DIRECTORY = str(Path(__file__).parent.joinpath("web"))


def _load_model_downloader():
    model_downloader_path = Path(__file__).parent / "Model-Downloader" / "__init__.py"
    if not model_downloader_path.exists():
        return

    spec = spec_from_file_location(
        "link_comfy_nodes.model_downloader", model_downloader_path
    )
    if not spec or not spec.loader:
        return

    try:
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"Failed to load Model-Downloader: {exc}")


_load_model_downloader()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
