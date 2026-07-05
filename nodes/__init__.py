"""Node implementations grouped by domain."""

from .color import ColorParserNode, FarthestColorNode, MatchColorPaletteNode
from .image import (
    AddImageToBatchNode,
    AutoCropperNode,
    CropByBBoxNode,
    CropToContentNode,
    ImageRotatorNode,
    LoadFolderNode,
    PixelationDimensionsNode,
    PixPunkRemoveBackground,
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
    TrimConditioning,
    VACESampler,
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
    "PixPunkRemoveBackground",
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
    "ImageRotatorNode",
    "KSamplerAdvancedDual",
    "LoadFolderNode",
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
    "SpriteScaleCalculatorNode",
    "SaveFolderAsZip",
    "SaveImageSequenceZip",
    "SpritesheetBuilderNode",
    "SpritesheetPreviewNode",
    "TemporalMaskCropper",
    "TrimConditioning",
    "VACESampler",
    "ShiftImageBatchNode",
    "WANFrameCalculatorNode",
    "WANFramesToAddAndCut",
]
