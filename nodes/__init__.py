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
    BatchMaskCropper,
    ReplaceAlpha,
    SEGSFlatten,
    StabilizerTrimNode,
    TemporalMaskCropper,
    TrimConditioning,
    VideoDetailer,
    VideoMaskEditor,
    VideoTileDetailer,
)
from .wan import (
    LoadVACEModuleNode,
    NativeWanPoseStrength,
    WANFrameCalculatorNode,
    WANFramesToAddAndCut,
)

__all__ = [
    "AddImageToBatchNode",
    "AdvancedStringConcat",
    "BatchMaskCropper",
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
    "LoadVACEModuleNode",
    "MatchColorPaletteNode",
    "NativeWanPoseStrength",
    "PixelationDimensionsNode",
    "PoseImageSetupNode",
    "PreviewAsMarkdown",
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
    "VideoDetailer",
    "VideoMaskEditor",
    "WANFrameCalculatorNode",
    "WANFramesToAddAndCut",
]
