"""Node implementations grouped by domain."""

from .color import (
    ApplyPaletteNode,
    ColorParserNode,
    FarthestColorNode,
    MatchColorsToReferenceNode,
)
from .image import (
    AddImageToBatchNode,
    AutoCropperNode,
    CropByBBoxNode,
    ImagePixelateNode,
    ImageRotatorNode,
    LoadFolderNode,
    ResizeImageAndMaskBySideNode,
    SpriteScaleCalculatorNode,
    SpritesheetBuilderNode,
)
from .latent import SnapToDivisible
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
    SpritesheetPreviewNode,
)
from .sampling import KSamplerAdvancedDual
from .save import BatchImageSave, SaveFolderAsZip, SaveImageSequenceZip
from .text import AdvancedStringConcat, DropdownSelectNode, PreviewAsMarkdown, StringToListNode
from .video import (
    AverageMaskRegionSizeNode,
    ReplaceAlpha,
)
from .wan import (
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
    "FarthestColorNode",
    "FastImagePreviewNode",
    "ImageCompareNode",
    "ImagePixelateNode",
    "ImageRotatorNode",
    "KSamplerAdvancedDual",
    "LoadFolderNode",
    "NativeWanPoseStrength",
    "PreviewAsMarkdown",
    "StringToListNode",
    "PreviewImageAlpha",
    "ReplaceAlpha",
    "ResizeImageAndMaskBySideNode",
    "SpriteScaleCalculatorNode",
    "SaveFolderAsZip",
    "SaveImageSequenceZip",
    "SpritesheetBuilderNode",
    "SpritesheetPreviewNode",
    "ShiftImageBatchNode",
    "WANFrameCalculatorNode",
    "WANFramesToAddAndCut",
]
