import math


class WANFrameCalculatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10000,
                        "step": 1,
                        "display": "number",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("wan_frames",)
    FUNCTION = "calculate_wan_frames"
    CATEGORY = "animation/utils"

    def calculate_wan_frames(self, frame_count):
        """
        Calculate the nearest WAN-compatible frame count (at or above input).
        WAN accepts frames following the pattern: 1, 5, 9, 13, 17, 21, ...
        Formula: 1 + (x * 4) where x >= 0

        Args:
            frame_count: Input frame count to round up to WAN-compatible value

        Returns:
            Nearest WAN-compatible frame count (>= input)
        """
        if frame_count <= 1:
            wan_frames = 1
            x = 0
        else:
            # Calculate x needed: frame_count <= 1 + (x * 4)
            # Solving for x: x >= (frame_count - 1) / 4
            # We need to round up to get the nearest value at or above frame_count
            x = math.ceil((frame_count - 1) / 4)
            wan_frames = 1 + (x * 4)

        print(f"Input frames: {frame_count} → WAN frames: {wan_frames} (x={x})")

        return (wan_frames,)
