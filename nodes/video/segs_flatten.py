"""Fix 5D SEGS tensors for Impact Pack compatibility (e.g. after video detailer)."""

import torch
import torch.nn.functional as F


class SEGSFlatten:
    """Flatten 5D cropped_image tensors inside SEGS to 4D (NHWC) so Impact Pack
    nodes like SEGSPaste don't raise 'Expected NHWC tensor' errors.

    Also resizes cropped_image spatial dims to match cropped_mask when they differ
    (e.g. when an upscale-detailer enhanced at a different resolution than the crop)."""

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
            mask = seg.cropped_mask

            # 1. Flatten 5D (B, F, H, W, C) → 4D (B*F, H, W, C)
            if img is not None and img.ndim == 5:
                b, f, h, w, c = img.shape
                img = img.reshape(b * f, h, w, c)

            # 2. Resize image spatial dims to match mask if they differ
            if img is not None and mask is not None:
                mask_h = mask.shape[-2]
                mask_w = mask.shape[-1]
                img_h = img.shape[1]
                img_w = img.shape[2]

                if img_h != mask_h or img_w != mask_w:
                    # NHWC → NCHW for interpolate, then back
                    img_nchw = img.permute(0, 3, 1, 2).float()
                    img_nchw = F.interpolate(
                        img_nchw, size=(mask_h, mask_w), mode="bilinear", align_corners=False
                    )
                    img = img_nchw.permute(0, 2, 3, 1).to(seg.cropped_image.dtype)

            fixed.append(
                SEG(
                    img,
                    mask,
                    seg.confidence,
                    seg.crop_region,
                    seg.bbox,
                    seg.label,
                    seg.control_net_wrapper,
                )
            )

        return ((segs[0], fixed),)
