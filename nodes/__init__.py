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
    NormalizeSpriteEntityHeightNode,
    SpritesheetBuilderNode,
)
from .latent import AspectToResolution, SnapToDivisible
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
    ShiftPoseFramesNode,
    ShiftImageBatchNode,
    UnshiftPoseFramesNode,
    WANFrameCalculatorNode,
    WANFramesToAddAndCut,
)

__all__ = [
    "AddImageToBatchNode",
    "ApplyPaletteNode",
    "ShiftPoseFramesNode",
    "AdvancedStringConcat",
    "AverageMaskRegionSizeNode",
    "DropdownSelectNode",
    "AspectToResolution",
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
    "NormalizeSpriteEntityHeightNode",
    "SaveFolderAsZip",
    "SaveImageSequenceZip",
    "SpritesheetBuilderNode",
    "SpritesheetPreviewNode",
    "ShiftImageBatchNode",
    "UnshiftPoseFramesNode",
    "WANFrameCalculatorNode",
    "WANFramesToAddAndCut",
]
