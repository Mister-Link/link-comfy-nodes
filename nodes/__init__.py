"""Node implementations grouped by domain."""

from .auto_cropper import AutoCropperNode
from .bgeraser_nodes import BulkBackgroundRemoverBgEraserNode
from .color_nodes import ColorParserNode, FarthestColorNode, MatchColorPaletteNode
from .detailer_animatediff_nag import DetailerForAnimateDiffNAG
from .fast_image_preview import FastImagePreviewNode
from .image_nodes import (
    AddImageToBatchNode,
    CropToContentNode,
    ImageRotatorNode,
    PixelationDimensionsNode,
    PoseImageSetupNode,
    ResizeImageAndMaskBySideNode,
    SpritesheetBuilderNode,
)
from .nag_detailer_hook import NAGCompatibleDetailerHook
from .native_wan import NativeWanPoseStrength
from .pixel_art.node import ConvertToPixelArt
from .segs_fixer import (
    DetailerForEachPipeForAnimateDiffFixed,
    SEGSFixCropRegionForNAGNode,
    SEGSFixDimensionsNode,
)
from .spritesheet_preview import SpritesheetPreviewNode
from .stabilizer_trim import StabilizerTrimNode
from .string_nodes import AdvancedStringConcat, PreviewAsMarkdown, SaveFolderAsZip
from .video_nodes import (
    BatchImageSave,
    PreviewImageAlpha,
    ReplaceAlpha,
    SaveImageSequenceZip,
    VideoMaskEditor,
    WANFrameCalculatorNode,
)
from .wan_frame_adjuster import WANFramesToAddAndCut

__all__ = [
    "AddImageToBatchNode",
    "ColorParserNode",
    "FarthestColorNode",
    "MatchColorPaletteNode",
    "FastImagePreviewNode",
    "ImageRotatorNode",
    "BulkBackgroundRemoverBgEraserNode",
    "CropToContentNode",
    "PixelationDimensionsNode",
    "PoseImageSetupNode",
    "ResizeImageAndMaskBySideNode",
    "SpritesheetBuilderNode",
    "SpritesheetPreviewNode",
    "ConvertToPixelArt",
    "AdvancedStringConcat",
    "PreviewAsMarkdown",
    "SaveFolderAsZip",
    "AutoCropperNode",
    "BatchImageSave",
    "PreviewImageAlpha",
    "ReplaceAlpha",
    "SaveImageSequenceZip",
    "StabilizerTrimNode",
    "VideoMaskEditor",
    "NativeWanPoseStrength",
    "WANFrameCalculatorNode",
    "WANFramesToAddAndCut",
    "SEGSFixDimensionsNode",
    "SEGSFixCropRegionForNAGNode",
    "DetailerForEachPipeForAnimateDiffFixed",
    "NAGCompatibleDetailerHook",
    "DetailerForAnimateDiffNAG",
]
