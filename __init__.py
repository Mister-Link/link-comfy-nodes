from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from .nodes import (
    AddImageToBatchNode,
    AdvancedStringConcat,
    StringToListNode,
    AutoCropperNode,
    PixPunkRemoveBackground,
    AverageMaskRegionSizeNode,
    BatchImageSave,
    ChangeLatentDimensions,
    ColorParserNode,
    ConvertToPixelArt,
    CropByBBoxNode,
    CropToContentNode,
    FarthestColorNode,
    FastImagePreviewNode,
    ImageRotatorNode,
    KSamplerAdvancedDual,
    LoadVACEModuleNode,
    MatchColorPaletteNode,
    NativeWanPoseStrength,
    PixelationDimensionsNode,
    PreviewAsMarkdown,
    PreviewImageAlpha,
    PreviewWebmNode,
    ReplaceAlpha,
    ResizeImageAndMaskBySideNode,
    SaveFolderAsZip,
    SaveImageSequenceZip,
    SnapToDivisible,
    SpritesheetBuilderNode,
    SpritesheetPreviewNode,
    TemporalMaskCropper,
    TrimConditioning,
    VACESampler,
    LoopSCAILPoseFramesNode,
    ShiftImageBatchNode,
    WANFrameCalculatorNode,
    WANFramesToAddAndCut,
)

NODE_CLASS_MAPPINGS = {
    "Load VACE Module": LoadVACEModuleNode,
    "Average Mask Region Size": AverageMaskRegionSizeNode,
    "Add Image to Batch": AddImageToBatchNode,
    "Change Latent Dimensions": ChangeLatentDimensions,
    "Snap to Divisible": SnapToDivisible,
    "Hex or 24 Bit Color": ColorParserNode,
    "Farthest Color": FarthestColorNode,
    "Match Color Palette": MatchColorPaletteNode,
    "Loop SCAIL Pose Frames": LoopSCAILPoseFramesNode,
    "WAN Frame Calculator": WANFrameCalculatorNode,
    "Native Wan Pose Strength": NativeWanPoseStrength,
    "Image Rotator": ImageRotatorNode,
    "KSampler Advanced (Dual Output)": KSamplerAdvancedDual,
    "Crop by BBox": CropByBBoxNode,
    "Crop to Content": CropToContentNode,
    "Pixelation Dimensions": PixelationDimensionsNode,
    "Resize Image and Mask by Side": ResizeImageAndMaskBySideNode,
    "Spritesheet Builder": SpritesheetBuilderNode,
    "Spritesheet Preview": SpritesheetPreviewNode,
    "PreviewImageAlpha": PreviewImageAlpha,
    "ReplaceAlpha": ReplaceAlpha,
    "Save To Zip": SaveImageSequenceZip,
    "ConvertToPixelArt": ConvertToPixelArt,
    "BatchImageSave": BatchImageSave,
    "Concat Strings": AdvancedStringConcat,
    "String to List": StringToListNode,
    "Save Folder as ZIP": SaveFolderAsZip,
    "PreviewAsMarkdown": PreviewAsMarkdown,
    "Auto Cropper": AutoCropperNode,
    "Fast Image Preview": FastImagePreviewNode,
    "Temporal Mask Cropper": TemporalMaskCropper,
    "Trim Conditioning": TrimConditioning,
    "Shift Image Batch": ShiftImageBatchNode,
    "VACE Sampler": VACESampler,
    "WAN Frames to Add & Cut": WANFramesToAddAndCut,
    "Preview (webm)": PreviewWebmNode,
    "Remove Background": PixPunkRemoveBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load VACE Module": "Load VACE Module",
    "Average Mask Region Size": "Average Mask Region Size",
    "Add Image to Batch": "Add Image to Batch",
    "Change Latent Dimensions": "Change Latent Dimensions",
    "Snap to Divisible": "Snap to Divisible",
    "Hex or 24 Bit Color": "Convert Color Format",
    "Farthest Color": "Find Furthest Color",
    "Match Color Palette": "Match Color Palette",
    "Loop SCAIL Pose Frames": "Loop SCAIL Pose Frames",
    "WAN Frame Calculator": "WAN Frame Calculator",
    "Native Wan Pose Strength": "Native Wan Pose Strength",
    "Image Rotator": "Rotate Image",
    "KSampler Advanced (Dual Output)": "KSampler Advanced (Dual Output)",
    "Crop by BBox": "Crop by BBox",
    "Crop to Content": "Crop to Content",
    "Pixelation Dimensions": "Pixelation Dimensions",
    "Resize Image and Mask by Side": "Resize Image and Mask by Side",
    "Spritesheet Builder": "Spritesheet Builder",
    "Spritesheet Preview": "Spritesheet Preview",
    "PreviewImageAlpha": "Preview Image (Alpha)",
    "ReplaceAlpha": "Replace Alpha",
    "Save To Zip": "Save to ZIP",
    "ConvertToPixelArt": "Convert to Pixel Art",
    "BatchImageSave": "Batch Image Save",
    "Concat Strings": "Concat Strings",
    "String to List": "String to List",
    "Save Folder as ZIP": "Save Folder as ZIP",
    "PreviewAsMarkdown": "Preview as Markdown",
    "Auto Cropper": "Auto Cropper",
    "Fast Image Preview": "Fast Image Preview",
    "Temporal Mask Cropper": "Temporal Mask Cropper",
    "Trim Conditioning": "Trim Conditioning",
    "Shift Image Batch": "Shift Image Batch",
    "VACE Sampler": "VACE Sampler",
    "WAN Frames to Add & Cut": "WAN Frames to Add & Cut",
    "Preview (webm)": "Preview (webm)",
    "Remove Background": "Remove Background",
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
