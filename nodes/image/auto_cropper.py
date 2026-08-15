from __future__ import annotations

import cv2
import json
import numpy as np
import torch


class AutoCropperNode:
    CATEGORY = "image/transform"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("cropped_frames", "cropped_alpha", "bbox")
    FUNCTION = "auto_crop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "method": (
                    [
                        "bg_sub",
                        "shared_bg_sub",
                        "trim_bars",
                    ],
                    {"default": "bg_sub"},
                ),
                "sensitivity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "padding": (
                    "INT",
                    {"default": 5, "min": 0, "max": 500, "step": 1},
                ),
                "padding_color": (
                    "STRING",
                    {"default": "#000000"},
                ),
                "use_image_padding": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If true, padding area is taken from the original frame when in bounds; "
                            "out-of-bounds area uses padding_color."
                        ),
                    },
                ),
                "pad_edge_pixel": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If true, padding is filled with the averaged color along each edge of "
                            "the crop (not the literal edge pixels -- replicating exact pixels "
                            "streaks if the edge has any per-pixel noise or dither). Overrides "
                            "padding_color and use_image_padding when enabled."
                        ),
                    },
                ),
            },
            "optional": {
                "alpha": ("MASK",),
            },
        }

    def _detect_background_subtraction(self, frame_np, sensitivity):
        h, w = frame_np.shape[:2]
        rgb = frame_np[:, :, :3] if frame_np.shape[2] >= 3 else frame_np

        sensitivity = float(np.clip(sensitivity, 0.0, 1.0))

        bg_color = self._estimate_corner_background(rgb)

        return self._detect_background_from_color(rgb, bg_color, sensitivity)

    def _estimate_corner_background(self, rgb):
        h, w = rgb.shape[:2]
        corner_size = max(2, int(min(h, w) * 0.05))
        corners = [
            rgb[:corner_size, :corner_size],
            rgb[:corner_size, -corner_size:],
            rgb[-corner_size:, :corner_size],
            rgb[-corner_size:, -corner_size:],
        ]
        corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
        return np.median(corner_pixels, axis=0)

    def _estimate_batch_corner_background(self, frames_np):
        rgb_frames = frames_np[..., :3] if frames_np.shape[-1] >= 3 else frames_np
        if rgb_frames.ndim != 4:
            raise ValueError(
                f"Expected frames with shape (N, H, W, C), got {rgb_frames.shape}"
            )

        _, h, w, _ = rgb_frames.shape
        corner_size = max(2, int(min(h, w) * 0.05))
        corner_pixels = []
        for frame in rgb_frames:
            corner_pixels.extend(
                [
                    frame[:corner_size, :corner_size].reshape(-1, 3),
                    frame[:corner_size, -corner_size:].reshape(-1, 3),
                    frame[-corner_size:, :corner_size].reshape(-1, 3),
                    frame[-corner_size:, -corner_size:].reshape(-1, 3),
                ]
            )
        return np.median(np.vstack(corner_pixels), axis=0)

    def _foreground_threshold(self, sensitivity, low, high):
        sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        return low + (sensitivity * (high - low))

    def _detect_background_from_color(self, rgb, bg_color, sensitivity):
        sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        diff = np.linalg.norm(rgb - bg_color[None, None, :], axis=2)

        # Low sensitivity picks up smaller deviations from the shared background.
        base_threshold = self._foreground_threshold(sensitivity, 0.02, 0.25)
        fg_mask = (diff > base_threshold).astype(np.uint8) * 255

        return self._cleanup_foreground_mask(fg_mask)

    def _cleanup_foreground_mask(self, fg_mask):
        # Clean up: close small holes, erode to remove noise, dilate to restore size
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_k)

        # Erode to remove thin noise
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        fg_mask = cv2.erode(fg_mask, erode_k, iterations=1)

        # Dilate to restore edges and connect small gaps
        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.dilate(fg_mask, dilate_k, iterations=2)

        return fg_mask

    def _detect_shared_background_subtraction(self, frame_np, shared_bg_color, sensitivity):
        rgb = frame_np[:, :, :3] if frame_np.shape[2] >= 3 else frame_np
        sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        diff = np.linalg.norm(rgb - shared_bg_color[None, None, :], axis=2)

        # Shared-background crops are aimed at stable greenscreen/uniform-bg
        # sequences, so they intentionally use a stricter threshold range than
        # per-frame bg_sub. This avoids background spill/noise in one frame
        # inflating the union crop for the whole batch.
        base_threshold = self._foreground_threshold(
            sensitivity, 40.0 / 255.0, 100.0 / 255.0
        )
        return (diff > base_threshold).astype(np.uint8) * 255

    @staticmethod
    def _detect_bar_extent(rgb_u8, edge, tolerance):
        # Counts how many rows/columns from `edge` are part of a solid bar,
        # stopping at the first one that isn't. Makes no assumption about
        # bar thickness -- only that the bar starts exactly at the edge,
        # which is what a letterbox/pillarbox bar is.
        #
        # A row/column only counts as "bar" if it passes two checks: it
        # must be internally uniform (every pixel in it close to that
        # row/column's own mean), AND that mean must match the sampled
        # border color. Matching the border color alone isn't enough --
        # on an edge with no actual bar, a row/column can average out
        # close to the border color just by blending unrelated regions
        # (e.g. a column that's part white bar and part green content
        # averages to something that spuriously resembles other such
        # columns), which reads as a bar where there isn't one. Requiring
        # uniformity first rules that out: a blended column has high
        # internal deviation regardless of what its average happens to be.
        h, w, _ = rgb_u8.shape

        if edge in ("top", "bottom"):
            border_color = rgb_u8[0 if edge == "top" else -1].mean(axis=0)
            indices = range(h) if edge == "top" else range(h - 1, -1, -1)
            line = lambda i: rgb_u8[i]
        else:
            border_color = rgb_u8[:, 0 if edge == "left" else -1].mean(axis=0)
            indices = range(w) if edge == "left" else range(w - 1, -1, -1)
            line = lambda i: rgb_u8[:, i]

        count = 0
        for i in indices:
            pixels = line(i).astype(np.float64)
            mean_color = pixels.mean(axis=0)
            internal_deviation = np.abs(pixels - mean_color).sum(axis=-1).max()
            border_diff = np.abs(mean_color - border_color).sum()
            if internal_deviation <= tolerance and border_diff <= tolerance:
                count += 1
            else:
                break
        return count

    def _detect_trim_bars(self, frame_np, sensitivity):
        # Solid-color letterbox/pillarbox bars, trimmed independently of any
        # single global "background color": each edge is checked against
        # its own sampled color, inward row by row (or column by column),
        # so this only ever removes genuine full-width/height bars --
        # unlike bg_sub, it can't be thrown off by interior content that
        # happens to be close in color to the bar, or by a corner sample
        # that happens to land on content instead of the bar.
        frame_uint8 = (frame_np * 255).astype(np.uint8)
        rgb = frame_uint8[:, :, :3] if frame_uint8.shape[2] >= 3 else frame_uint8
        h, w = rgb.shape[:2]

        sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        # Higher sensitivity = stricter match required to call a row/column
        # part of the bar (trims less); lower sensitivity is more
        # forgiving of a noisy/gradient bar (trims more). Tuned against a
        # real solid bar: genuine bar rows differed from the sampled bar
        # color by at most ~5; the antialiased seam row jumped to ~60+.
        tolerance = 5.0 + (1.0 - sensitivity) * 45.0
        feather = 1  # also trim the ~1px antialiased seam, not just the flat bar

        mask = np.full((h, w), 255, dtype=np.uint8)
        for edge in ("top", "bottom", "left", "right"):
            extent = self._detect_bar_extent(rgb, edge, tolerance)
            if extent <= 0:
                continue
            extent = min(extent + feather, h if edge in ("top", "bottom") else w)
            if edge == "top":
                mask[:extent, :] = 0
            elif edge == "bottom":
                mask[h - extent :, :] = 0
            elif edge == "left":
                mask[:, :extent] = 0
            else:
                mask[:, w - extent :] = 0
        return mask

    def _mask_to_bbox(self, mask):
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        return x, y, x + w, y + h

    def _mask_to_bbox_line_filtered(self, mask, min_line_pixels=4):
        foreground = mask > 0
        row_counts = foreground.sum(axis=1)
        col_counts = foreground.sum(axis=0)
        rows = np.flatnonzero(row_counts >= min_line_pixels)
        cols = np.flatnonzero(col_counts >= min_line_pixels)
        if rows.size == 0 or cols.size == 0:
            return None
        return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1

    @staticmethod
    def _edge_average_pad(image, top, bottom, left, right):
        # cv2.copyMakeBorder(..., BORDER_REPLICATE) repeats the literal
        # edge row/column outward. That's fine for a perfectly flat edge,
        # but any per-pixel noise or dither pattern along that edge gets
        # stretched into the padding wholesale, which reads as visible
        # streaking (each noisy column/row keeps its own slightly-off
        # value, repeated many times, instead of blending away). Filling
        # each padding band with the *average* color along that edge
        # avoids this: the padding is flat and streak-free regardless of
        # how noisy the source edge is.
        h, w = image.shape[:2]
        out_shape = (h + top + bottom, w + left + right) + image.shape[2:]
        out = np.empty(out_shape, dtype=image.dtype)
        out[top : top + h, left : left + w] = image

        top_avg = image[0].mean(axis=0) if top > 0 else None
        bottom_avg = image[-1].mean(axis=0) if bottom > 0 else None
        left_avg = image[:, 0].mean(axis=0) if left > 0 else None
        right_avg = image[:, -1].mean(axis=0) if right > 0 else None

        if top > 0:
            out[:top, left : left + w] = top_avg
        if bottom > 0:
            out[top + h :, left : left + w] = bottom_avg
        if left > 0:
            out[top : top + h, :left] = left_avg
        if right > 0:
            out[top : top + h, left + w :] = right_avg

        # Corners: blend the two adjacent edge averages so there's no seam
        # between (e.g.) the top band's color and the left band's color.
        if top > 0 and left > 0:
            out[:top, :left] = (top_avg + left_avg) / 2
        if top > 0 and right > 0:
            out[:top, left + w :] = (top_avg + right_avg) / 2
        if bottom > 0 and left > 0:
            out[top + h :, :left] = (bottom_avg + left_avg) / 2
        if bottom > 0 and right > 0:
            out[top + h :, left + w :] = (bottom_avg + right_avg) / 2

        return out

    def _hex_to_rgb(self, hex_color: str):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r, g, b = (
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16),
            )
            return (r / 255.0, g / 255.0, b / 255.0)
        return (0.0, 0.0, 0.0)

    def auto_crop(
        self,
        frames: torch.Tensor,
        method: str,
        sensitivity: float,
        padding: int,
        padding_color: str,
        use_image_padding: bool,
        pad_edge_pixel: bool = False,
        alpha: torch.Tensor = None,
    ):
        print(f"[AutoCropper] Processing {frames.shape[0]} frames using {method}...")

        # Backward compatibility for older workflow values.
        if method == "background_subtraction":
            method = "bg_sub"
        elif method in ("sequence_bg_sub", "temporal_bg_sub", "shared_background"):
            method = "shared_bg_sub"
        elif method in ("letterbox", "pillarbox", "trim", "bars"):
            method = "trim_bars"

        frames_np = frames.cpu().numpy()

        if frames_np.ndim != 4:
            raise ValueError(
                f"Expected frames with shape (N, H, W, C), got {frames_np.shape}"
            )

        num_frames, H, W, C = frames_np.shape

        if alpha is not None:
            alpha_np = alpha.cpu().numpy()
            if alpha_np.ndim == 4 and alpha_np.shape[-1] == 1:
                alpha_np = alpha_np[..., 0]
            if alpha_np.ndim != 3:
                raise ValueError(
                    f"Expected alpha with shape (N, H, W), got {alpha_np.shape}"
                )
            if alpha_np.shape[1:3] != (H, W):
                print(
                    f"[AutoCropper] alpha resolution {alpha_np.shape[1:3]} does not "
                    f"match frames resolution {(H, W)}; resizing alpha to match."
                )
                resized = np.empty((alpha_np.shape[0], H, W), dtype=alpha_np.dtype)
                for i in range(alpha_np.shape[0]):
                    resized[i] = cv2.resize(
                        alpha_np[i], (W, H), interpolation=cv2.INTER_LINEAR
                    )
                alpha_np = resized
            if alpha_np.shape[0] != num_frames:
                if alpha_np.shape[0] == 1:
                    alpha_np = np.repeat(alpha_np, num_frames, axis=0)
                else:
                    raise ValueError(
                        f"alpha has {alpha_np.shape[0]} frames but frames has "
                        f"{num_frames}; frame counts must match (or alpha must have 1 frame)"
                    )
        else:
            alpha_np = np.ones((num_frames, H, W), dtype=np.float32)

        print(f"[AutoCropper] Analyzing frames with sensitivity {sensitivity}...")
        global_box = None
        shared_bg_color = None

        if method == "shared_bg_sub":
            shared_bg_color = self._estimate_batch_corner_background(frames_np)
            print(
                "[AutoCropper] Using shared background color "
                f"{np.round(shared_bg_color, 4).tolist()} across {num_frames} frames"
            )

        for i, frame in enumerate(frames_np):
            if method == "bg_sub":
                mask = self._detect_background_subtraction(frame, sensitivity)
                bbox = self._mask_to_bbox(mask)
            elif method == "shared_bg_sub":
                mask = self._detect_shared_background_subtraction(
                    frame, shared_bg_color, sensitivity
                )
                bbox = self._mask_to_bbox_line_filtered(mask)
            elif method == "trim_bars":
                mask = self._detect_trim_bars(frame, sensitivity)
                bbox = self._mask_to_bbox(mask)
            else:
                raise ValueError(f"Unknown method: {method}")

            if bbox is None:
                continue

            if global_box is None:
                global_box = list(bbox)
            else:
                global_box[0] = min(global_box[0], bbox[0])
                global_box[1] = min(global_box[1], bbox[1])
                global_box[2] = max(global_box[2], bbox[2])
                global_box[3] = max(global_box[3], bbox[3])

            if (i + 1) % 10 == 0:
                print(f"[AutoCropper] Processed {i + 1}/{num_frames} frames")

        if global_box is None:
            raise RuntimeError("No content detected in any frame")

        x1, y1, x2, y2 = global_box

        crop_width = x2 - x1
        crop_height = y2 - y1

        print(
            f"[AutoCropper] Content bounds: ({x1}, {y1}) -> ({x2}, {y2}), "
            f"size: {crop_width}x{crop_height}"
        )

        cropped_frames = []
        cropped_alphas = []

        for i in range(num_frames):
            frame = frames_np[i]
            alpha_frame = alpha_np[i]

            cropped_frame = frame[y1:y2, x1:x2]
            cropped_alpha = alpha_frame[y1:y2, x1:x2]

            if padding > 0:
                padded_h = crop_height + (padding * 2)
                padded_w = crop_width + (padding * 2)

                if use_image_padding:
                    src_x1 = x1 - padding
                    src_y1 = y1 - padding
                    src_x2 = x2 + padding
                    src_y2 = y2 + padding

                    if pad_edge_pixel:
                        # Pull real image pixels first; once the original frame's
                        # edge is reached, extend the outermost real pixel outward
                        # to fill the remaining out-of-bounds padding.
                        border_left = max(0, -src_x1)
                        border_top = max(0, -src_y1)
                        border_right = max(0, src_x2 - W)
                        border_bottom = max(0, src_y2 - H)

                        ext_frame = self._edge_average_pad(
                            frame, border_top, border_bottom, border_left, border_right
                        )
                        ext_alpha = self._edge_average_pad(
                            alpha_frame, border_top, border_bottom, border_left, border_right
                        )

                        ex1 = src_x1 + border_left
                        ey1 = src_y1 + border_top
                        padded_frame = ext_frame[
                            ey1 : ey1 + padded_h, ex1 : ex1 + padded_w
                        ]
                        padded_alpha = ext_alpha[
                            ey1 : ey1 + padded_h, ex1 : ex1 + padded_w
                        ]
                    else:
                        r, g, b = self._hex_to_rgb(padding_color)

                        padded_frame = np.zeros(
                            (padded_h, padded_w, C), dtype=cropped_frame.dtype
                        )
                        if C >= 3:
                            padded_frame[:, :, 0] = r
                            padded_frame[:, :, 1] = g
                            padded_frame[:, :, 2] = b
                        if C == 4:
                            padded_frame[:, :, 3] = 0.0

                        padded_alpha = np.zeros(
                            (padded_h, padded_w), dtype=cropped_alpha.dtype
                        )

                        # Copy only the in-bounds part from the original frame.
                        ix1 = max(0, src_x1)
                        iy1 = max(0, src_y1)
                        ix2 = min(W, src_x2)
                        iy2 = min(H, src_y2)

                        if ix2 > ix1 and iy2 > iy1:
                            dx1 = ix1 - src_x1
                            dy1 = iy1 - src_y1
                            dx2 = dx1 + (ix2 - ix1)
                            dy2 = dy1 + (iy2 - iy1)
                            padded_frame[dy1:dy2, dx1:dx2] = frame[iy1:iy2, ix1:ix2]
                            padded_alpha[dy1:dy2, dx1:dx2] = alpha_frame[iy1:iy2, ix1:ix2]
                elif pad_edge_pixel:
                    padded_frame = self._edge_average_pad(
                        cropped_frame, padding, padding, padding, padding
                    )
                    padded_alpha = self._edge_average_pad(
                        cropped_alpha, padding, padding, padding, padding
                    )
                else:
                    r, g, b = self._hex_to_rgb(padding_color)

                    padded_frame = np.zeros(
                        (padded_h, padded_w, C), dtype=cropped_frame.dtype
                    )
                    if C >= 3:
                        padded_frame[:, :, 0] = r
                        padded_frame[:, :, 1] = g
                        padded_frame[:, :, 2] = b
                    if C == 4:
                        padded_frame[:, :, 3] = 0.0

                    padded_alpha = np.zeros(
                        (padded_h, padded_w), dtype=cropped_alpha.dtype
                    )

                    padded_frame[
                        padding : padding + crop_height, padding : padding + crop_width
                    ] = cropped_frame
                    padded_alpha[
                        padding : padding + crop_height, padding : padding + crop_width
                    ] = cropped_alpha

                cropped_frames.append(padded_frame)
                cropped_alphas.append(padded_alpha)
            else:
                cropped_frames.append(cropped_frame)
                cropped_alphas.append(cropped_alpha)

        result_frames = torch.from_numpy(np.stack(cropped_frames).astype(np.float32))
        result_alphas = torch.from_numpy(np.stack(cropped_alphas).astype(np.float32))

        final_width = result_frames.shape[2]
        final_height = result_frames.shape[1]

        print(
            f"[AutoCropper] Done. Output size: {final_width}x{final_height} "
            f"(with {padding}px padding)"
        )

        bbox_data = {
            "x": int(x1),
            "y": int(y1),
            "w": int(crop_width),
            "h": int(crop_height),
            "width": int(final_width),
            "height": int(final_height),
        }
        bbox_json = json.dumps(bbox_data)

        return (result_frames, result_alphas, bbox_json)
