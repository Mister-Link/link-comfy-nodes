"""Node implementations grouped by domain."""

from .color import ColorParserNode, FarthestColorNode, MatchColorPaletteNode
from .image import (
    AddImageToBatchNode,
    AutoCropperNode,
    LocalBackgroundRemoverNode,
    CropByBBoxNode,
    ImageRotatorNode,
    PixelationDimensionsNode,
    ResizeImageAndMaskBySideNode,
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
    TrimConditioning,
    VACESampler,
)
from .wan import (
    LoadVACEModuleNode,
    NativeWanPoseStrength,
    LoopSCAILPoseFramesNode,
    WANFrameCalculatorNode,
    WANFramesToAddAndCut,
)

__all__ = [
    "AddImageToBatchNode",
    "LoopSCAILPoseFramesNode",
    "AdvancedStringConcat",
    "AverageMaskRegionSizeNode",
    "ChangeLatentDimensions",
    "SnapToDivisible",
    "AutoCropperNode",
    "BatchImageSave",
    "LocalBackgroundRemoverNode",
    "ColorParserNode",
    "ConvertToPixelArt",
    "CropByBBoxNode",
    "FarthestColorNode",
    "FastImagePreviewNode",
    "ImageRotatorNode",
    "KSamplerAdvancedDual",
    "LoadVACEModuleNode",
    "MatchColorPaletteNode",
    "NativeWanPoseStrength",
    "PixelationDimensionsNode",
    "PreviewAsMarkdown",
    "StringToListNode",
    "PreviewImageAlpha",
    "PreviewWebmNode",
    "ReplaceAlpha",
    "ResizeImageAndMaskBySideNode",
    "SaveFolderAsZip",
    "SaveImageSequenceZip",
    "SpritesheetBuilderNode",
    "SpritesheetPreviewNode",
    "TemporalMaskCropper",
    "TrimConditioning",
    "VACESampler",
    "WANFrameCalculatorNode",
    "WANFramesToAddAndCut",
]
