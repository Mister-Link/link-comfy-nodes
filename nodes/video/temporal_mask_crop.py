"""Temporal mask cropper - crops all frames to the union bounds across the batch."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TemporalMaskCropper:
    """Crop all frames to a single temporally stable bounding box.

    By default, bounds come from the union of the input mask across the batch.
    Optionally, the crop region can be detected from the image content using the
    same detection modes exposed by Auto Cropper.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "method": (
                    [
                        "mask",
                        "bbox",
                        "anime",
                        "bg_sub",
                    ],
                    {"default": "mask"},
                ),
                "sensitivity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "padding": ("INT", {"default": 16, "min": 0, "max": 512}),
                "use_image_padding": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "crop"
    CATEGORY = "link/video"
    DESCRIPTION = (
        "Crops all frames to one temporally stable bounding box across the batch. "
        "method=mask uses the input mask and sensitivity as the mask threshold. "
        "Other methods mirror Auto Cropper detection modes and use sensitivity for detection. "
        "use_image_padding=True extends the crop window into surrounding image content "
        "(clamped at edges). use_image_padding=False fills padding with black."
    )

    def _normalize_method(self, method: str) -> str:
        if method == "anime_seg":
            return "anime"
        if method == "background_subtraction":
            return "bg_sub"
        if method in ("bbox_detection", "alpha_channel", "contour_detection"):
            return "bbox"
        return method

    def _prepare_masks(self, masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
        masks = masks.float()
        if masks.shape[1] != height or masks.shape[2] != width:
            masks = F.interpolate(
                masks.unsqueeze(1), size=(height, width), mode="nearest-exact"
            ).squeeze(1)
        return masks

    def _get_union_bbox_from_mask(
        self, masks: torch.Tensor, sensitivity: float
    ) -> tuple[int, int, int, int] | None:
        union_mask = masks.max(dim=0).values
        y_idx, x_idx = torch.nonzero(union_mask > sensitivity, as_tuple=True)
        if y_idx.numel() == 0:
            return None
        return (
            int(x_idx.min().item()),
            int(y_idx.min().item()),
            int(x_idx.max().item()) + 1,
            int(y_idx.max().item()) + 1,
        )

    def _get_union_bbox_from_images(
        self, images: torch.Tensor, method: str, sensitivity: float
    ) -> tuple[int, int, int, int] | None:
        from ..image.auto_cropper import AutoCropperNode

        detector = AutoCropperNode()

        if method == "anime":
            model, device = detector._get_model()
        else:
            model, device = None, None

        global_box = None

        for frame in images.detach().cpu().numpy():
            if method == "anime":
                detection_mask = detector._segment_frame_anime(
                    frame, model, device, sensitivity
                )
            elif method == "bg_sub":
                detection_mask = detector._detect_background_subtraction(
                    frame, sensitivity
                )
            elif method == "bbox":
                detection_mask = detector._detect_bbox(frame, sensitivity)
            else:
                raise ValueError(f"Unknown method: {method}")

            bbox = detector._mask_to_bbox(detection_mask)
            if bbox is None:
                continue

            if global_box is None:
                global_box = list(bbox)
            else:
                global_box[0] = min(global_box[0], bbox[0])
                global_box[1] = min(global_box[1], bbox[1])
                global_box[2] = max(global_box[2], bbox[2])
                global_box[3] = max(global_box[3], bbox[3])

        return tuple(global_box) if global_box is not None else None

    def _crop_to_bbox(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        bbox: tuple[int, int, int, int],
        padding: int,
        use_image_padding: bool,
    ):
        frame_count, height, width, _ = images.shape
        mask_count = masks.shape[0]
        x1, y1, x2, y2 = bbox

        if use_image_padding:
            min_y = max(0, y1 - padding)
            max_y = min(height, y2 + padding)
            min_x = max(0, x1 - padding)
            max_x = min(width, x2 + padding)

            cropped_images = images[:, min_y:max_y, min_x:max_x, :]
            if mask_count == 1:
                cropped_masks = masks[0:1, min_y:max_y, min_x:max_x].expand(
                    frame_count, -1, -1
                )
            else:
                cropped_masks = masks[:, min_y:max_y, min_x:max_x]

            crop_h = max_y - min_y
            crop_w = max_x - min_x
        else:
            cropped_images = images[:, y1:y2, x1:x2, :]
            cropped_images = F.pad(
                cropped_images.permute(0, 3, 1, 2),
                (padding, padding, padding, padding),
                mode="constant",
                value=0,
            ).permute(0, 2, 3, 1)

            if mask_count == 1:
                tight_masks = masks[0:1, y1:y2, x1:x2]
            else:
                tight_masks = masks[:, y1:y2, x1:x2]

            cropped_masks = F.pad(
                tight_masks,
                (padding, padding, padding, padding),
                mode="constant",
                value=0,
            )
            if mask_count == 1:
                cropped_masks = cropped_masks.expand(frame_count, -1, -1)

            crop_h = cropped_images.shape[1]
            crop_w = cropped_images.shape[2]

        return cropped_images, cropped_masks, crop_w, crop_h

    def crop(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        method: str = "mask",
        sensitivity: float = 0.5,
        padding: int = 16,
        use_image_padding: bool = True,
    ):
        """Crop all frames to a shared bounding box.

        Args:
            images: (T, H, W, C) batch of frames
            masks:  (T, H, W) or (1, H, W) batch of masks (values 0–1)
            method: crop-region detection source; mask uses the provided mask,
                other values mirror Auto Cropper image detection methods
            sensitivity: threshold/detection sensitivity for the selected method
            padding: extra pixels around the final bounding box
            use_image_padding: if True, padding region uses real image pixels;
                if False, padding is filled with black
        """
        frame_count, height, width, _ = images.shape
        mask_count = masks.shape[0]
        method = self._normalize_method(method)
        sensitivity = float(max(0.0, min(1.0, sensitivity)))

        masks = self._prepare_masks(masks, height, width)

        if method == "mask":
            bbox = self._get_union_bbox_from_mask(masks, sensitivity)
        else:
            bbox = self._get_union_bbox_from_images(images, method, sensitivity)

        if bbox is None:
            print(
                "[TemporalMaskCropper] no crop content found, returning full frames "
                f"(method={method}, sensitivity={sensitivity})"
            )
            return (images, masks if mask_count == frame_count else masks.expand(frame_count, height, width))

        cropped_images, cropped_masks, crop_w, crop_h = self._crop_to_bbox(
            images=images,
            masks=masks,
            bbox=bbox,
            padding=padding,
            use_image_padding=use_image_padding,
        )

        print(
            f"[TemporalMaskCropper] → {crop_w}×{crop_h} "
            f"(method={method}, sensitivity={sensitivity}, padding={padding}, "
            f"use_image_padding={use_image_padding}, frames={frame_count})"
        )

        return (cropped_images, cropped_masks)
