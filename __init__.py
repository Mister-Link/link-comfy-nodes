"""
@author: Mister-Link
@title: Link Comfy Nodes
@nickname: LinkNodes
@version: 1.0.0
@project: "https://github.com/Mister-Link/link-comfy-nodes",
@description: Just some helpful nodes!
"""

from .color_parser import ColorParserNode  # assuming your node file is color_parser.py

NODE_CLASS_MAPPINGS = {
    "Hex or 24 Bit Color": ColorParserNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hex or 24 Bit Color": "Hex / 24-bit Color Parser",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
