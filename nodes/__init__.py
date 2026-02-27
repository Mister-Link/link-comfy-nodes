"""Node implementations grouped by domain."""

from .color import ColorParserNode, FarthestColorNode, MatchColorPaletteNode
from .image import (
    AddImageToBatchNode,
    AutoCropperNode,
    BulkBackgroundRemoverBgEraserNode,
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
    VaceControlStrength,
    WANFrameCalculatorNode,
    WANFramesToAddAndCut,
)

__all__ = [
    "AddImageToBatchNode",
    "AdvancedStringConcat",
    "AutoCropperNode",
    "BatchImageSave",
    "BulkBackgroundRemoverBgEraserNode",
    "ColorParserNode",
    "ConvertToPixelArt",
    "CropToContentNode",
    "FarthestColorNode",
    "FastImagePreviewNode",
    "ImageRotatorNode",
    "MatchColorPaletteNode",
    "NativeWanPoseStrength",
    "VaceControlStrength",
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
