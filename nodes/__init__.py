"""Node implementations grouped by domain."""

from .color import ColorParserNode, FarthestColorNode, MatchColorPaletteNode
from .image import (
    AddImageToBatchNode,
    AutoCropperNode,
    BulkBackgroundRemoverBgEraserNode,
    CropByBBoxNode,
    CropToContentNode,
    ImageRotatorNode,
    PixelationDimensionsNode,
    PoseImageSetupNode,
    ResizeImageAndMaskBySideNode,
    SpritesheetBuilderNode,
)
from .pixel_art.node import ConvertToPixelArt
from .preview import (
    FastImagePreviewNode,
    PreviewImageAlpha,
    PreviewWebmNode,
    SpritesheetPreviewNode,
)
from .latent import ChangeLatentDimensions, SnapToDivisible
from .sampling import KSamplerAdvancedDual
from .save import BatchImageSave, SaveFolderAsZip, SaveImageSequenceZip
from .text import AdvancedStringConcat, PreviewAsMarkdown
from .video import (
    ReplaceAlpha,
    StabilizerTrimNode,
    TrimConditioning,
    VideoDetailer,
    VideoMaskEditor,
)
from .wan import (
    NativeWanPoseStrength,
    WANFrameCalculatorNode,
    WANFramesToAddAndCut,
)

__all__ = [
    "AddImageToBatchNode",
    "AdvancedStringConcat",
    "ChangeLatentDimensions",
    "SnapToDivisible",
    "AutoCropperNode",
    "BatchImageSave",
    "BulkBackgroundRemoverBgEraserNode",
    "ColorParserNode",
    "ConvertToPixelArt",
    "CropByBBoxNode",
    "CropToContentNode",
    "FarthestColorNode",
    "FastImagePreviewNode",
    "ImageRotatorNode",
    "KSamplerAdvancedDual",
    "MatchColorPaletteNode",
    "NativeWanPoseStrength",
    "PixelationDimensionsNode",
    "PoseImageSetupNode",
    "PreviewAsMarkdown",
    "PreviewImageAlpha",
    "PreviewWebmNode",
    "ReplaceAlpha",
    "ResizeImageAndMaskBySideNode",
    "SaveFolderAsZip",
    "SaveImageSequenceZip",
    "SpritesheetBuilderNode",
    "SpritesheetPreviewNode",
    "StabilizerTrimNode",
    "TrimConditioning",
    "VideoDetailer",
    "VideoMaskEditor",
    "WANFrameCalculatorNode",
    "WANFramesToAddAndCut",
]
