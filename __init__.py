from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from .nodes import (
    AddImageToBatchNode,
    AdvancedStringConcat,
    AutoCropperNode,
    BatchImageSave,
    BulkBackgroundRemoverBgEraserNode,
    ColorParserNode,
    CropToContentNode,
    DetailerForEachPipeForAnimateDiffFixed,
    FarthestColorNode,
    FastImagePreviewNode,
    ImageRotatorNode,
    MatchColorPaletteNode,
    NativeWanPoseStrength,
    PixelationDimensionsNode,
    PoseImageSetupNode,
    PreviewAsMarkdown,
    PreviewImageAlpha,
    ReplaceAlpha,
    ResizeImageAndMaskBySideNode,
    SaveFolderAsZip,
    SaveImageSequenceZip,
    SEGSFixCropRegionForNAGNode,
    SEGSFixDimensionsNode,
    SpritesheetBuilderNode,
    SpritesheetPreviewNode,
    StabilizerTrimNode,
    VideoMaskEditor,
    WANFrameCalculatorNode,
    WANFramesToAddAndCut,
)
from .nodes.pixel_art.node import ConvertToPixelArt
from .nodes.simple_video_preview import PreviewAnimation

NODE_CLASS_MAPPINGS = {
    "Add Image to Batch": AddImageToBatchNode,
    "Hex or 24 Bit Color": ColorParserNode,
    "Farthest Color": FarthestColorNode,
    "Match Color Palette": MatchColorPaletteNode,
    "WAN Frame Calculator": WANFrameCalculatorNode,
    "Native Wan Pose Strength": NativeWanPoseStrength,
    "Image Rotator": ImageRotatorNode,
    "Remove Background": BulkBackgroundRemoverBgEraserNode,
    "Crop to Content": CropToContentNode,
    "Pixelation Dimensions": PixelationDimensionsNode,
    "Pose Image Setup": PoseImageSetupNode,
    "Resize Image and Mask by Side": ResizeImageAndMaskBySideNode,
    "Spritesheet Builder": SpritesheetBuilderNode,
    "Spritesheet Preview": SpritesheetPreviewNode,
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
    "Fast Image Preview": FastImagePreviewNode,
    "Stabilizer Trim": StabilizerTrimNode,
    "SEGS Fix Dimensions": SEGSFixDimensionsNode,
    "SEGS Fix Crop Region for NAG": SEGSFixCropRegionForNAGNode,
    "DetailerForEachPipe (AnimateDiff) Fixed": DetailerForEachPipeForAnimateDiffFixed,
    "PreviewAnimation": PreviewAnimation,
    "WAN Frames to Add & Cut": WANFramesToAddAndCut,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Add Image to Batch": "Add Image to Batch",
    "Hex or 24 Bit Color": "Convert Color Format",
    "Farthest Color": "Find Furthest Color",
    "Match Color Palette": "Match Color Palette",
    "WAN Frame Calculator": "WAN Frame Calculator",
    "Native Wan Pose Strength": "Native Wan Pose Strength",
    "Image Rotator": "Rotate Image",
    "Remove Background": "Remove Background",
    "Crop to Content": "Crop to Content",
    "Pixelation Dimensions": "Pixelation Dimensions",
    "Pose Image Setup": "Pose Image Setup",
    "Resize Image and Mask by Side": "Resize Image and Mask by Side",
    "Spritesheet Builder": "Spritesheet Builder",
    "Spritesheet Preview": "Spritesheet Preview",
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
    "Fast Image Preview": "Fast Image Preview",
    "Stabilizer Trim": "Stabilizer Trim",
    "SEGS Fix Dimensions": "SEGS Fix Dimensions",
    "SEGS Fix Crop Region for NAG": "SEGS Fix Crop Region for NAG",
    "DetailerForEachPipe (AnimateDiff) Fixed": "DetailerForEachPipe (AnimateDiff) Fixed",
    "PreviewAnimation": "Preview Animation",
    "WAN Frames to Add & Cut": "WAN Frames to Add & Cut",
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


# Auto-patch Impact Pack for 5D tensor support
def _apply_impact_pack_patches():
    """Apply compatibility patches to Impact Pack on startup."""
    try:
        from pathlib import Path

        install_script = Path(__file__).parent / "install.py"
        if install_script.exists():
            import subprocess

            result = subprocess.run(
                ["python3", str(install_script)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print("[link-comfy-nodes] Impact Pack patches applied successfully")
            else:
                print(f"[link-comfy-nodes] Patch script warning: {result.stdout}")
    except Exception as exc:
        print(f"[link-comfy-nodes] Could not apply patches: {exc}")


_apply_impact_pack_patches()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
