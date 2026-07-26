"""Node implementations grouped by domain."""

from .color import (
    ApplyPaletteNode,
    ColorParserNode,
    FarthestColorNode,
    MatchColorPaletteNode,
    MatchColorsToReferenceNode,
)
from .image import (
    AddImageToBatchNode,
    AutoCropperNode,
    CropByBBoxNode,
    CropToContentNode,
    ImagePixelateNode,
    ImageRotatorNode,
    LoadFolderNode,
    ResizeImageAndMaskBySideNode,
    SpriteScaleCalculatorNode,
    SpritesheetBuilderNode,
)
from .latent import ChangeLatentDimensions, SnapToDivisible
from .motion import (
    HybridMaMoMaskExportFBX,
    HybridMaMoMaskGenerate,
    HybridMaMoMaskLoader,
    HybridMaMoMaskPreviewAnimation,
)
from .pixel_art.node import ConvertToPixelArt
from .pixel_art.studio import PixelArtStudioNode
from .preview import (
    FastImagePreviewNode,
    ImageCompareNode,
    PreviewImageAlpha,
    PreviewWebmNode,
    SpritesheetPreviewNode,
)
from .sampling import KSamplerAdvancedDual
from .save import BatchImageSave, SaveFolderAsZip, SaveImageSequenceZip
from .text import AdvancedStringConcat, DropdownSelectNode, PreviewAsMarkdown, StringToListNode
from .video import (
    AverageMaskRegionSizeNode,
    ReplaceAlpha,
    TemporalMaskCropper,
)
from .wan import (
    LoadVACEModuleNode,
    NativeWanPoseStrength,
    LoopSCAILPoseFramesNode,
    ShiftImageBatchNode,
    WANFrameCalculatorNode,
    WANFramesToAddAndCut,
)

__all__ = [
    "AddImageToBatchNode",
    "ApplyPaletteNode",
    "LoopSCAILPoseFramesNode",
    "AdvancedStringConcat",
    "AverageMaskRegionSizeNode",
    "DropdownSelectNode",
    "ChangeLatentDimensions",
    "SnapToDivisible",
    "HybridMaMoMaskLoader",
    "HybridMaMoMaskGenerate",
    "HybridMaMoMaskPreviewAnimation",
    "HybridMaMoMaskExportFBX",
    "AutoCropperNode",
    "BatchImageSave",
    "ColorParserNode",
    "ConvertToPixelArt",
    "CropByBBoxNode",
    "CropToContentNode",
    "FarthestColorNode",
    "FastImagePreviewNode",
    "ImageCompareNode",
    "ImagePixelateNode",
    "ImageRotatorNode",
    "KSamplerAdvancedDual",
    "LoadFolderNode",
    "LoadVACEModuleNode",
    "MatchColorPaletteNode",
    "NativeWanPoseStrength",
    "PreviewAsMarkdown",
    "StringToListNode",
    "PreviewImageAlpha",
    "PreviewWebmNode",
    "ReplaceAlpha",
    "ResizeImageAndMaskBySideNode",
    "SpriteScaleCalculatorNode",
    "SaveFolderAsZip",
    "SaveImageSequenceZip",
    "SpritesheetBuilderNode",
    "SpritesheetPreviewNode",
    "TemporalMaskCropper",
    "ShiftImageBatchNode",
    "WANFrameCalculatorNode",
    "WANFramesToAddAndCut",
]
