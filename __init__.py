from pathlib import Path

from .nodes import (
    ColorParserNode,
    FarthestColorNode,
    ImageRotatorNode,
    PoseImageSetupNode,
    VideoMaskEditor,
    WANFrameCalculatorNode,
)

NODE_CLASS_MAPPINGS = {
    "Hex or 24 Bit Color": ColorParserNode,
    "Farthest Color": FarthestColorNode,
    "WAN Frame Calculator": WANFrameCalculatorNode,
    "Image Rotator": ImageRotatorNode,
    "Pose Image Setup": PoseImageSetupNode,
    "VideoMaskEditor": VideoMaskEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hex or 24 Bit Color": "Convert Color Format",
    "Farthest Color": "Find Furthest Color",
    "WAN Frame Calculator": "WAN Frame Calculator",
    "Image Rotator": "Rotate Image",
    "Pose Image Setup": "Pose Image Setup",
    "VideoMaskEditor": "Video Mask Editor",
}

WEB_DIRECTORY = str(Path(__file__).parent.joinpath("web"))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
