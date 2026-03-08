"""Fix 5D SEGS tensors for Impact Pack compatibility (e.g. after video detailer)."""


class SEGSFlatten:
    """Flatten 5D cropped_image tensors inside SEGS to 4D (NHWC) so Impact Pack
    nodes like SEGSPaste don't raise 'Expected NHWC tensor' errors."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segs": ("SEGS",)}}

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = "Video/SEGS"

    def doit(self, segs):
        try:
            from impact.core import SEG
        except ImportError:
            from modules.impact.core import SEG

        fixed = []
        for seg in segs[1]:
            img = seg.cropped_image
            if img is not None and img.ndim == 5:
                b, f, h, w, c = img.shape
                img = img.reshape(b * f, h, w, c)
            fixed.append(
                SEG(
                    img,
                    seg.cropped_mask,
                    seg.confidence,
                    seg.crop_region,
                    seg.bbox,
                    seg.label,
                    seg.control_net_wrapper,
                )
            )

        return ((segs[0], fixed),)
