"""
@author: Mister-Link
@title: Link Comfy Nodes
@nickname: LinkNodes
@version: 1.0.0
@project: "https://github.com/Mister-Link/link-comfy-nodes",
@description: A collection of custom ComfyUI nodes by Mister-Link, including color parsing and utility nodes.
"""

from .furthest_color import FarthestColorNode
from .string_to_color import ColorParserNode
from .wan_frame_calculator import WANFrameCalculatorNode

NODE_CLASS_MAPPINGS = {
    "Hex or 24 Bit Color": ColorParserNode,
    "Farthest Color": FarthestColorNode,
    "WAN Frame Calculator": WANFrameCalculatorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hex or 24 Bit Color": "Convert Color Format",
    "Farthest Color": "Find Furthest Color",
    "WAN Frame Calculator": "WAN Frame Calculator",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
