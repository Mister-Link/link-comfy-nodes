"""
Monkey patch for ComfyUI-SAM3 to fix frame_idx float->int conversion issue.

Issue: ComfyUI-SAM3's SAM3VideoSegmentation.segment() receives frame_idx as a float
(e.g., 0.3) from ComfyUI, but the SAM3 model expects an integer index for list access.

This causes: TypeError: list indices must be integers or slices, not float

Solution: Wrap the segment method to ensure frame_idx is converted to int before processing.
"""


def patch_sam3_video_segmentation():
    """
    Monkey patch SAM3VideoSegmentation.segment to convert frame_idx to int.
    """
    try:
        # Try to import the SAM3 nodes
        import sys
        from pathlib import Path

        # Find ComfyUI-SAM3 in custom_nodes
        custom_nodes_path = Path(__file__).parent.parent
        sam3_path = custom_nodes_path / "ComfyUI-SAM3" / "nodes"

        if not sam3_path.exists():
            print("[SAM3 Patch] ComfyUI-SAM3 not found, skipping patch")
            return

        # Add to path if not already there
        sam3_path_str = str(sam3_path.parent)
        if sam3_path_str not in sys.path:
            sys.path.insert(0, sam3_path_str)

        # Import the module
        try:
            from nodes import sam3_video_nodes
        except ImportError:
            print("[SAM3 Patch] Could not import sam3_video_nodes, skipping patch")
            return

        # Get the class
        if not hasattr(sam3_video_nodes, "SAM3VideoSegmentation"):
            print("[SAM3 Patch] SAM3VideoSegmentation class not found, skipping patch")
            return

        SAM3VideoSegmentation = sam3_video_nodes.SAM3VideoSegmentation

        # Store original method
        original_segment = SAM3VideoSegmentation.segment

        # Create wrapper that ensures frame_idx is int
        def patched_segment(
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
            """Patched segment method that ensures frame_idx is an integer."""
            # Convert frame_idx to int (fix for float values from ComfyUI)
            frame_idx = int(frame_idx)

            # Call original method with converted frame_idx
            return original_segment(
                self,
                video_frames,
                prompt_mode,
                text_prompt,
                positive_points,
                negative_points,
                positive_boxes,
                negative_boxes,
                frame_idx,
                score_threshold,
            )

        # Apply the patch
        SAM3VideoSegmentation.segment = patched_segment

        print(
            "[SAM3 Patch] Successfully patched SAM3VideoSegmentation.segment to handle float frame_idx"
        )

    except Exception as e:
        print(f"[SAM3 Patch] Failed to apply patch: {e}")
        import traceback

        traceback.print_exc()


def apply_patches():
    """Apply all SAM3 patches."""
    patch_sam3_video_segmentation()


# Auto-apply on import
if __name__ != "__main__":
    apply_patches()
