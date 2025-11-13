"""
@author: Mister-Link
@title: Link Comfy Nodes
@nickname: LinkNodes
@version: 1.0.0
@project: "https://github.com/Mister-Link/link-comfy-nodes",
@description: A collection of custom ComfyUI nodes by Mister-Link, including color parsing and utility nodes.
"""

from .string_to_color import ColorParserNode

NODE_CLASS_MAPPINGS = {
    "Hex or 24 Bit Color": ColorParserNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hex or 24 Bit Color": "Hex / 24-bit Color Parser",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
