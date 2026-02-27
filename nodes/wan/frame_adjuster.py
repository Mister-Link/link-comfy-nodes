class WANFramesToAddAndCut:
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
        def is_valid_wan_count(n):
            if n < 1:
                return False
            return (n - 1) % 4 == 0

        def get_next_valid_wan_count(target):
            if target < 1:
                return 1
            n = (target - 1) // 4
            result = n * 4 + 1
            if result < target:
                result = (n + 1) * 4 + 1
            return result

        def get_prev_valid_wan_count(target):
            if target < 1:
                return 1
            n = (target - 1) // 4
            result = n * 4 + 1
            return max(1, result)

        if not is_valid_wan_count(frame_count):
            frame_count = get_next_valid_wan_count(frame_count)

        if frames_to_add == 0:
            valid_add_options = [0]
        else:
            next_add = get_next_valid_wan_count(frames_to_add)
            prev_add = get_prev_valid_wan_count(frames_to_add)

            if abs(next_add - frames_to_add) <= abs(prev_add - frames_to_add):
                valid_add_options = [next_add, prev_add]
            else:
                valid_add_options = [prev_add, next_add]

        for test_add in valid_add_options:
            target_result = frame_count + test_add - frames_to_cut
            candidates = []

            if target_result >= 1 and is_valid_wan_count(target_result):
                candidates.append(
                    (test_add, frames_to_cut, abs(test_add - frames_to_add))
                )

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

            if candidates:
                candidates.sort(key=lambda x: x[2])
                best = candidates[0]
                return (frame_count, best[0], best[1])

        for test_add_n in range(20):
            test_add = test_add_n * 4 + 1 if test_add_n > 0 else 0

            for cut_delta in range(-20, 20):
                test_cut = max(0, frames_to_cut + cut_delta)
                result = frame_count + test_add - test_cut

                if (
                    result >= 1
                    and is_valid_wan_count(result)
                    and test_cut < frame_count + test_add
                ):
                    return (frame_count, test_add, test_cut)

        return (frame_count, 0, 0)
