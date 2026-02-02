"""
SAM3 Video Segmentation Wrapper Node

This is a wrapper around ComfyUI-SAM3's SAM3VideoSegmentation node that fixes
the frame_idx float->int conversion issue.

Use this node instead of the original SAM3VideoSegmentation to avoid the
"TypeError: list indices must be integers or slices, not float" error.
"""


class SAM3VideoSegmentationFixed:
    """
    Wrapper for SAM3VideoSegmentation that ensures frame_idx is an integer.

    This fixes the issue where ComfyUI passes frame_idx as a float (e.g., 0.3)
    which causes a TypeError when SAM3 tries to use it as a list index.
    """

    PROMPT_MODES = ["text", "point", "box"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": (
                    "IMAGE",
                    {"tooltip": "Video frames as batch of images [N, H, W, C]"},
                ),
                "prompt_mode": (
                    cls.PROMPT_MODES,
                    {
                        "default": "text",
                        "tooltip": "Prompt type: text (describe objects), point (click on objects), or box (draw rectangles)",
                    },
                ),
            },
            "optional": {
                # Text mode inputs
                "text_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "[text mode] Text description(s) to track. Comma-separated for multiple objects (e.g., 'person, dog, car')",
                    },
                ),
                # Point mode inputs
                "positive_points": (
                    "SAM3_POINTS_PROMPT",
                    {
                        "tooltip": "[point mode] Positive points - click on objects to track"
                    },
                ),
                "negative_points": (
                    "SAM3_POINTS_PROMPT",
                    {
                        "tooltip": "[point mode] Negative points - click on areas to exclude"
                    },
                ),
                # Box mode inputs
                "positive_boxes": (
                    "SAM3_BOXES_PROMPT",
                    {
                        "tooltip": "[box mode] Positive boxes - draw around objects to track"
                    },
                ),
                "negative_boxes": (
                    "SAM3_BOXES_PROMPT",
                    {
                        "tooltip": "[box mode] Negative boxes - draw around areas to exclude"
                    },
                ),
                # Common inputs
                "frame_idx": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": "Frame index to apply prompts (usually 0 for first frame)",
                    },
                ),
                "score_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Detection confidence threshold",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SAM3_VIDEO_STATE",)
    RETURN_NAMES = ("video_state",)
    FUNCTION = "segment"
    CATEGORY = "SAM3/video"

    def segment(
        self,
        video_frames,
        prompt_mode="text",
        text_prompt="",
        positive_points=None,
        negative_points=None,
        positive_boxes=None,
        negative_boxes=None,
        frame_idx=0,
        score_threshold=0.3,
    ):
        """
        Wrapper that ensures frame_idx is an integer before calling the original node.
        """
        # Import the original SAM3 node
        try:
            import sys
            from pathlib import Path

            # Find and import the SAM3 module
            custom_nodes_path = Path(__file__).parent.parent.parent
            sam3_module_path = custom_nodes_path / "ComfyUI-SAM3"

            if str(sam3_module_path) not in sys.path:
                sys.path.insert(0, str(sam3_module_path))

            from nodes.sam3_video_nodes import SAM3VideoSegmentation

        except Exception as e:
            raise ImportError(
                f"Failed to import ComfyUI-SAM3. Make sure it's installed in custom_nodes. Error: {e}"
            )

        # Convert frame_idx to integer (this is the fix!)
        frame_idx = int(frame_idx)

        # Create an instance of the original node and call it
        original_node = SAM3VideoSegmentation()

        return original_node.segment(
            video_frames=video_frames,
            prompt_mode=prompt_mode,
            text_prompt=text_prompt,
            positive_points=positive_points,
            negative_points=negative_points,
            positive_boxes=positive_boxes,
            negative_boxes=negative_boxes,
            frame_idx=frame_idx,
            score_threshold=score_threshold,
        )


NODE_CLASS_MAPPINGS = {
    "SAM3VideoSegmentationFixed": SAM3VideoSegmentationFixed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3VideoSegmentationFixed": "SAM3 Video Segmentation (Fixed)",
}
