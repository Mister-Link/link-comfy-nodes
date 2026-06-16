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
        def next_valid(n):
            n = max(1, n)
            rem = (n - 1) % 4
            return n if rem == 0 else n + (4 - rem)

        def prev_valid(n):
            n = max(1, n)
            return n - (n - 1) % 4

        target = frame_count + frames_to_add - frames_to_cut

        lo = prev_valid(max(1, target))
        hi = next_valid(max(1, target))

        best = None
        for valid_target in sorted({lo, hi}, key=lambda v: abs(v - target)):
            delta = valid_target - target

            # Option A: absorb delta into frames_to_cut (cut fewer or more)
            add_a, cut_a = frames_to_add, frames_to_cut - delta
            # Option B: absorb delta into frames_to_add (add more or fewer)
            add_b, cut_b = frames_to_add + delta, frames_to_cut

            for add, cut in ((add_a, cut_a), (add_b, cut_b)):
                if add >= 0 and cut >= 0 and cut < frame_count + add:
                    cost = abs(add - frames_to_add) + abs(cut - frames_to_cut)
                    if best is None or cost < best[3]:
                        best = (valid_target, add, cut, cost)

        if best:
            return (best[0], best[1], best[2])
        return (frame_count + frames_to_add - frames_to_cut, frames_to_add, frames_to_cut)
