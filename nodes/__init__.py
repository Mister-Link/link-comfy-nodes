"""Node implementations grouped by domain."""

from .auto_cropper import AutoCropperNode
from .bgeraser_nodes import BulkBackgroundRemoverBgEraserNode
from .color_nodes import ColorParserNode, FarthestColorNode
from .image_nodes import (
    CropToContentNode,
    ImageRotatorNode,
    PixelationDimensionsNode,
    PoseImageSetupNode,
    ResizeImageAndMaskBySideNode,
    SpritesheetBuilderNode,
)
from .pixel_art.node import ConvertToPixelArt
from .string_nodes import AdvancedStringConcat, PreviewAsMarkdown, SaveFolderAsZip
from .video_nodes import (
    BatchImageSave,
    PreviewImageAlpha,
    ReplaceAlpha,
    SaveImageSequenceZip,
    VideoMaskEditor,
    WANFrameCalculatorNode,
)

__all__ = [
    "ColorParserNode",
    "FarthestColorNode",
    "ImageRotatorNode",
    "BulkBackgroundRemoverBgEraserNode",
    "CropToContentNode",
    "PixelationDimensionsNode",
    "PoseImageSetupNode",
    "ResizeImageAndMaskBySideNode",
    "SpritesheetBuilderNode",
    "ConvertToPixelArt",
    "AdvancedStringConcat",
    "PreviewAsMarkdown",
    "SaveFolderAsZip",
    "AutoCropperNode",
    "BatchImageSave",
    "PreviewImageAlpha",
    "ReplaceAlpha",
    "SaveImageSequenceZip",
    "VideoMaskEditor",
    "WANFrameCalculatorNode",
]
