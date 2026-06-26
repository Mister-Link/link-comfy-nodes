"""Node implementations grouped by domain."""

from .color import ColorParserNode, FarthestColorNode, MatchColorPaletteNode
from .image import (
    AddImageToBatchNode,
    AutoCropperNode,
    LocalBackgroundRemoverNode,
    CropByBBoxNode,
    CropToContentNode,
    ImageRotatorNode,
    PixelationDimensionsNode,
    PoseImageSetupNode,
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
    BatchMaskCropper,
    ReplaceAlpha,
    SEGSFlatten,
    StabilizerTrimNode,
    TemporalMaskCropper,
    TrimConditioning,
    VACESampler,
    VideoMaskEditor,
    VideoTileDetailer,
)
from .wan import (
    LoadVACEModuleNode,
    NativeWanPoseStrength,
    LoopSCAILPoseFramesNode,
    WanVideoAddSCAILPoseEmbedsMasked,
    WANFrameCalculatorNode,
    WANFramesToAddAndCut,
)

__all__ = [
    "AddImageToBatchNode",
    "LoopSCAILPoseFramesNode",
    "AdvancedStringConcat",
    "AverageMaskRegionSizeNode",
    "BatchMaskCropper",
    "ChangeLatentDimensions",
    "SnapToDivisible",
    "AutoCropperNode",
    "BatchImageSave",
    "LocalBackgroundRemoverNode",
    "ColorParserNode",
    "ConvertToPixelArt",
    "CropByBBoxNode",
    "CropToContentNode",
    "FarthestColorNode",
    "FastImagePreviewNode",
    "ImageRotatorNode",
    "KSamplerAdvancedDual",
    "LoadVACEModuleNode",
    "MatchColorPaletteNode",
    "NativeWanPoseStrength",
    "WanVideoAddSCAILPoseEmbedsMasked",
    "PixelationDimensionsNode",
    "PoseImageSetupNode",
    "PreviewAsMarkdown",
    "StringToListNode",
    "PreviewImageAlpha",
    "PreviewWebmNode",
    "ReplaceAlpha",
    "ResizeImageAndMaskBySideNode",
    "SEGSFlatten",
    "VideoTileDetailer",
    "SaveFolderAsZip",
    "SaveImageSequenceZip",
    "SpritesheetBuilderNode",
    "SpritesheetPreviewNode",
    "StabilizerTrimNode",
    "TemporalMaskCropper",
    "TrimConditioning",
    "VACESampler",
    "VideoMaskEditor",
    "WANFrameCalculatorNode",
    "WANFramesToAddAndCut",
]
