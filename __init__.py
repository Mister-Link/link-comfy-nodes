from pathlib import Path

from .nodes import (
    ColorParserNode,
    CropToContentNode,
    FarthestColorNode,
    ImageRotatorNode,
    PixelationDimensionsNode,
    PoseImageSetupNode,
    PreviewImageAlpha,
    ReplaceAlpha,
    SaveImageSequenceZip,
    VideoMaskEditor,
    WANFrameCalculatorNode,
)
from .nodes.pixel_art.node import ConvertToPixelArt

NODE_CLASS_MAPPINGS = {
    "Hex or 24 Bit Color": ColorParserNode,
    "Farthest Color": FarthestColorNode,
    "WAN Frame Calculator": WANFrameCalculatorNode,
    "Image Rotator": ImageRotatorNode,
    "Crop to Content": CropToContentNode,
    "Pixelation Dimensions": PixelationDimensionsNode,
    "Pose Image Setup": PoseImageSetupNode,
    "VideoMaskEditor": VideoMaskEditor,
    "PreviewImageAlpha": PreviewImageAlpha,
    "ReplaceAlpha": ReplaceAlpha,
    "SaveImageSequenceZip": SaveImageSequenceZip,
    "ConvertToPixelArt": ConvertToPixelArt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hex or 24 Bit Color": "Convert Color Format",
    "Farthest Color": "Find Furthest Color",
    "WAN Frame Calculator": "WAN Frame Calculator",
    "Image Rotator": "Rotate Image",
    "Crop to Content": "Crop to Content",
    "Pixelation Dimensions": "Pixelation Dimensions",
    "Pose Image Setup": "Pose Image Setup",
    "VideoMaskEditor": "Video Mask Editor",
    "PreviewImageAlpha": "Preview Image (Alpha)",
    "ReplaceAlpha": "Replace Alpha",
    "SaveImageSequenceZip": "Save Image Sequence (ZIP)",
    "ConvertToPixelArt": "Convert to Pixel Art",
}

WEB_DIRECTORY = str(Path(__file__).parent.joinpath("web"))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
