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
                "preference": (
                    ["balanced", "add frames", "remove frames"],
                    {"default": "balanced"},
                ),
            }
        }

    def calculate(
        self,
        frame_count: int,
        frames_to_add: int,
        frames_to_cut: int,
        preference: str,
    ):
        def next_valid(n: int) -> int:
            n = max(1, n)
            rem = (n - 1) % 4
            return n if rem == 0 else n + (4 - rem)

        def prev_valid(n: int) -> int:
            n = max(1, n)
            return n - ((n - 1) % 4)

        def candidate_frame_counts(target: int):
            lo = prev_valid(target)
            hi = next_valid(target)
            return sorted({lo, hi}, key=lambda value: (abs(value - target), value))

        requested_output = frame_count + frames_to_add - frames_to_cut

        # If the requested cut is large, we may need a larger add count just to keep
        # the final frame count positive. Search a bounded set of WAN-valid add
        # values that covers both the requested add count and that feasibility edge.
        min_add_for_positive_output = max(1, frames_to_cut - frame_count + 1)
        max_add_candidate = max(
            next_valid(frames_to_add),
            next_valid(min_add_for_positive_output),
        ) + 8

        best = None
        min_add_candidate = next_valid(max(1, frames_to_add))
        for add in range(min_add_candidate, max_add_candidate + 1, 4):
            target_output = frame_count + add - frames_to_cut

            for output_frame_count in candidate_frame_counts(target_output):
                cut = frame_count + add - output_frame_count
                if cut < 0:
                    continue
                if preference == "remove frames" and cut < frames_to_cut:
                    continue

                cost = abs(add - frames_to_add) + abs(cut - frames_to_cut)
                candidate = (
                    cost,
                    abs(output_frame_count - requested_output),
                    abs(add - frames_to_add),
                    abs(cut - frames_to_cut),
                    output_frame_count,
                    add,
                    cut,
                )

                if best is None or candidate < best:
                    best = candidate

        if best is None:
            fallback_add = min_add_candidate
            fallback_output = next_valid(
                max(1, frame_count + fallback_add - frames_to_cut)
            )
            fallback_cut = max(0, frame_count + fallback_add - fallback_output)
            if preference == "remove frames" and fallback_cut < frames_to_cut:
                fallback_cut = frames_to_cut
                fallback_output = next_valid(max(1, frame_count + fallback_add - fallback_cut))
            return (fallback_output, fallback_add, fallback_cut)

        return (best[4], best[5], best[6])
