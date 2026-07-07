"""Node implementations grouped by domain."""

from .color import ApplyPaletteNode, ColorParserNode, FarthestColorNode, MatchColorPaletteNode
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
from .pixel_art.node import ConvertToPixelArt
from .preview import (
    FastImagePreviewNode,
    PreviewImageAlpha,
    PreviewWebmNode,
    SpritesheetPreviewNode,
)
from .sampling import KSamplerAdvancedDual
from .save import BatchImageSave, SaveFolderAsZip, SaveImageSequenceZip
from .text import AdvancedStringConcat, PreviewAsMarkdown, StringToListNode
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
    "ChangeLatentDimensions",
    "SnapToDivisible",
    "AutoCropperNode",
    "BatchImageSave",
    "ColorParserNode",
    "ConvertToPixelArt",
    "CropByBBoxNode",
    "CropToContentNode",
    "FarthestColorNode",
    "FastImagePreviewNode",
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
