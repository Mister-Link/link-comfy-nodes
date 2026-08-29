from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from .nodes import (
    AddImageToBatchNode,
    AdvancedStringConcat,
    ApplyPaletteNode,
    StringToListNode,
    AutoCropperNode,
    AverageMaskRegionSizeNode,
    BatchImageSave,
    HybridMaMoMaskExportFBX,
    HybridMaMoMaskGenerate,
    HybridMaMoMaskLoader,
    HybridMaMoMaskPreviewAnimation,
    ColorParserNode,
    ConvertToPixelArt,
    PixelArtStudioNode,
    CropByBBoxNode,
    DropdownSelectNode,
    FarthestColorNode,
    FastImagePreviewNode,
    ImageCompareNode,
    ImagePixelateNode,
    ImageRotatorNode,
    KSamplerAdvancedDual,
    LoadFolderNode,
    MatchColorsToReferenceNode,
    NativeWanPoseStrength,
    PreviewAsMarkdown,
    PreviewImageAlpha,
    ReplaceAlpha,
    ResizeImageAndMaskBySideNode,
    SaveFolderAsZip,
    SaveImageSequenceZip,
    AspectToResolution,
    SnapToDivisible,
    SpritesheetBuilderNode,
    SpriteScaleCalculatorNode,
    StabilizeSpriteSequenceNode,
    SpritesheetPreviewNode,
    ShiftPoseFramesNode,
    ShiftImageBatchNode,
    UnshiftPoseFramesNode,
    WANFrameCalculatorNode,
    WANFramesToAddAndCut,
)

NODE_CLASS_MAPPINGS = {
    "Load Folder": LoadFolderNode,
    "Average Mask Region Size": AverageMaskRegionSizeNode,
    "Add Image to Batch": AddImageToBatchNode,
    "Palettize": ApplyPaletteNode,
    "HybridMaMoMask Loader": HybridMaMoMaskLoader,
    "HybridMaMoMask Generate": HybridMaMoMaskGenerate,
    "HybridMaMoMask Preview Animation (3D)": HybridMaMoMaskPreviewAnimation,
    "HybridMaMoMask Export FBX": HybridMaMoMaskExportFBX,
    "AspectToResolution": AspectToResolution,
    "Snap to Divisible": SnapToDivisible,
    "Hex or 24 Bit Color": ColorParserNode,
    "Farthest Color": FarthestColorNode,
    "Match Colors to Reference": MatchColorsToReferenceNode,
    "Shift Pose Frames": ShiftPoseFramesNode,
    "WAN Frame Calculator": WANFrameCalculatorNode,
    "Native Wan Pose Strength": NativeWanPoseStrength,
    "Pixelate": ImagePixelateNode,
    "Image Rotator": ImageRotatorNode,
    "KSampler Advanced (Dual Output)": KSamplerAdvancedDual,
    "Crop by BBox": CropByBBoxNode,
    "Resize Image and Mask by Side": ResizeImageAndMaskBySideNode,
    "Sprite Scale Calculator": SpriteScaleCalculatorNode,
    "Stabilize Sprite Sequence": StabilizeSpriteSequenceNode,
    "Spritesheet Builder": SpritesheetBuilderNode,
    "Spritesheet Preview": SpritesheetPreviewNode,
    "PreviewImageAlpha": PreviewImageAlpha,
    "ReplaceAlpha": ReplaceAlpha,
    "Save To Zip": SaveImageSequenceZip,
    "ConvertToPixelArt": ConvertToPixelArt,
    "Pixel Art Studio": PixelArtStudioNode,
    "BatchImageSave": BatchImageSave,
    "Concat Strings": AdvancedStringConcat,
    "String to List": StringToListNode,
    "Dropdown Select": DropdownSelectNode,
    "Save Folder as ZIP": SaveFolderAsZip,
    "PreviewAsMarkdown": PreviewAsMarkdown,
    "Auto Cropper": AutoCropperNode,
    "Fast Image Preview": FastImagePreviewNode,
    "Image Compare": ImageCompareNode,
    "Shift Image Batch": ShiftImageBatchNode,
    "Unshift Pose Frames": UnshiftPoseFramesNode,
    "WAN Frames to Add & Cut": WANFramesToAddAndCut,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Folder": "Load Folder",
    "Average Mask Region Size": "Average Mask Region Size",
    "Add Image to Batch": "Add Image to Batch",
    "Palettize": "Apply Game Palette",
    "HybridMaMoMask Loader": "HybridMaMoMask Loader",
    "HybridMaMoMask Generate": "HybridMaMoMask Generate",
    "HybridMaMoMask Preview Animation (3D)": "HybridMaMoMask Preview Animation (3D)",
    "HybridMaMoMask Export FBX": "HybridMaMoMask Export FBX",
    "AspectToResolution": "AspectToResolution",
    "Snap to Divisible": "Snap to Divisible",
    "Hex or 24 Bit Color": "Convert Color Format",
    "Farthest Color": "Find Furthest Color",
    "Match Colors to Reference": "Match Colors to Reference",
    "Shift Pose Frames": "Shift Pose Frames",
    "WAN Frame Calculator": "WAN Frame Calculator",
    "Native Wan Pose Strength": "Native Wan Pose Strength",
    "Pixelate": "Pixelate",
    "Image Rotator": "Rotate Image",
    "KSampler Advanced (Dual Output)": "KSampler Advanced (Dual Output)",
    "Crop by BBox": "Crop by BBox",
    "Resize Image and Mask by Side": "Resize Image and Mask by Side",
    "Sprite Scale Calculator": "Sprite Scale Calculator",
    "Stabilize Sprite Sequence": "Stabilize Sprite Sequence",
    "Spritesheet Builder": "Spritesheet Builder",
    "Spritesheet Preview": "Spritesheet Preview",
    "PreviewImageAlpha": "Preview Image (Alpha)",
    "ReplaceAlpha": "Replace Alpha",
    "Save To Zip": "Save to ZIP",
    "ConvertToPixelArt": "Convert to Pixel Art",
    "Pixel Art Studio": "Pixel Art Studio",
    "BatchImageSave": "Batch Image Save",
    "Concat Strings": "Concat Strings",
    "String to List": "String to List",
    "Dropdown Select": "Dropdown Select",
    "Save Folder as ZIP": "Save Folder as ZIP",
    "PreviewAsMarkdown": "Preview as Markdown",
    "Auto Cropper": "Auto Cropper",
    "Fast Image Preview": "Fast Image Preview",
    "Image Compare": "Image Compare",
    "Shift Image Batch": "Shift Image Batch",
    "Unshift Pose Frames": "Unshift Pose Frames",
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


def _apply_impact_pack_patches():
    install_script = Path(__file__).parent / "install.py"
    if not install_script.exists():
        return

    try:
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
