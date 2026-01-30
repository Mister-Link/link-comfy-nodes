"""
WAN Frames to Add & Cut node.

Calculates the closest values to frames_to_add and frames_to_cut while maintaining
the constraint that BOTH the output frame_count AND frames_to_add must follow the
n*4+1 formula (1, 5, 9, 13, etc.) and ensuring the result is non-negative.
"""


class WANFramesToAddAndCut:
    """
    WAN Frames to Add & Cut

    Adjusts frames_to_add and frames_to_cut to ensure:
    1. frames_to_add follows n*4+1 pattern (1, 5, 9, 13, 17, 21, etc.)
    2. The resulting frame_count follows n*4+1 pattern
    3. frames_to_cut < (frame_count + frames_to_add) to avoid negative results

    The algorithm prioritizes adjusting frames_to_add first, then frames_to_cut if needed.
    """

    CATEGORY = "conditioning/video_models"
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("frame_count", "frames_to_add", "frames_to_cut")
    FUNCTION = "calculate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": (
                    "INT",
                    {"default": 29, "min": 1, "max": 9999, "step": 1},
                ),
                "frames_to_add": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999, "step": 1},
                ),
                "frames_to_cut": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999, "step": 1},
                ),
            }
        }

    def calculate(self, frame_count: int, frames_to_add: int, frames_to_cut: int):
        """
        Calculate valid frames_to_add and frames_to_cut values.

        Args:
            frame_count: Current frame count (should already be n*4+1)
            frames_to_add: Desired frames to add (will be adjusted to n*4+1)
            frames_to_cut: Desired frames to cut

        Returns:
            Tuple of (frame_count, adjusted_frames_to_add, adjusted_frames_to_cut)
        """

        # Helper function to check if a number follows n*4+1 pattern
        def is_valid_wan_count(n):
            if n < 1:
                return False
            return (n - 1) % 4 == 0

        # Helper function to get nearest valid WAN count >= target
        def get_next_valid_wan_count(target):
            if target < 1:
                return 1
            n = (target - 1) // 4
            result = n * 4 + 1
            if result < target:
                result = (n + 1) * 4 + 1
            return result

        # Helper function to get nearest valid WAN count <= target
        def get_prev_valid_wan_count(target):
            if target < 1:
                return 1
            n = (target - 1) // 4
            result = n * 4 + 1
            return max(1, result)

        # Ensure frame_count is valid to start with
        if not is_valid_wan_count(frame_count):
            frame_count = get_next_valid_wan_count(frame_count)

        # First, adjust frames_to_add to nearest valid WAN count
        # Try both up and down, pick the closer one
        if frames_to_add == 0:
            valid_add_options = [0]  # 0 is allowed (no frames to add)
        else:
            next_add = get_next_valid_wan_count(frames_to_add)
            prev_add = get_prev_valid_wan_count(frames_to_add)

            # Pick the closer one, preferring upward if equal distance
            if abs(next_add - frames_to_add) <= abs(prev_add - frames_to_add):
                valid_add_options = [next_add, prev_add]
            else:
                valid_add_options = [prev_add, next_add]

        # Try each valid frames_to_add option
        for test_add in valid_add_options:
            # Now find frames_to_cut that makes the result valid
            # result = frame_count + test_add - frames_to_cut
            # We want: result to be n*4+1 and >= 1

            # Try adjusting frames_to_cut to make result valid
            target_result = frame_count + test_add - frames_to_cut

            # Try nearest valid results
            candidates = []

            # Option 1: Use desired cut if it gives valid result
            if target_result >= 1 and is_valid_wan_count(target_result):
                candidates.append(
                    (
                        test_add,
                        frames_to_cut,
                        abs(test_add - frames_to_add)
                        + abs(frames_to_cut - frames_to_cut),
                    )
                )

            # Option 2: Adjust cut to reach next valid result
            next_result = get_next_valid_wan_count(target_result)
            adjusted_cut_down = frames_to_cut - (next_result - target_result)
            if (
                adjusted_cut_down >= 0
                and frame_count + test_add - adjusted_cut_down >= 1
            ):
                candidates.append(
                    (
                        test_add,
                        adjusted_cut_down,
                        abs(test_add - frames_to_add)
                        + abs(adjusted_cut_down - frames_to_cut),
                    )
                )

            # Option 3: Adjust cut to reach previous valid result
            prev_result = get_prev_valid_wan_count(target_result)
            adjusted_cut_up = frames_to_cut + (target_result - prev_result)
            if adjusted_cut_up >= 0 and adjusted_cut_up < frame_count + test_add:
                result = frame_count + test_add - adjusted_cut_up
                if result >= 1 and is_valid_wan_count(result):
                    candidates.append(
                        (
                            test_add,
                            adjusted_cut_up,
                            abs(test_add - frames_to_add)
                            + abs(adjusted_cut_up - frames_to_cut),
                        )
                    )

            # If we found a valid candidate, return the best one
            if candidates:
                candidates.sort(key=lambda x: x[2])  # Sort by total delta
                best = candidates[0]
                return (frame_count, best[0], best[1])

        # Fallback: Try a broader search
        for test_add_n in range(20):  # Try n from 0 to 19
            test_add = test_add_n * 4 + 1 if test_add_n > 0 else 0

            # Try different cut values
            for cut_delta in range(-20, 20):
                test_cut = max(0, frames_to_cut + cut_delta)
                result = frame_count + test_add - test_cut

                if (
                    result >= 1
                    and is_valid_wan_count(result)
                    and test_cut < frame_count + test_add
                ):
                    return (frame_count, test_add, test_cut)

        # Last resort: no change
        return (frame_count, 0, 0)
