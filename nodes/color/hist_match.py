from __future__ import annotations

import cv2
import numpy as np
import torch

EPS = 1e-6


def _auto_foreground_mask(bgr: np.ndarray, threshold: float) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = bgr.shape[:2]
    patch = max(2, min(h, w) // 50)
    corners = np.concatenate((
        lab[:patch, :patch].reshape(-1, 3),
        lab[:patch, w - patch:].reshape(-1, 3),
        lab[h - patch:, :patch].reshape(-1, 3),
        lab[h - patch:, w - patch:].reshape(-1, 3),
    ), axis=0)
    background = np.median(corners, axis=0)
    mask = (np.linalg.norm(lab - background, axis=2) > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def _frame_stats(rgb: np.ndarray, mask_mode: str, threshold: float):
    bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = _auto_foreground_mask(bgr, threshold) if mask_mode == "auto" else np.ones(lab.shape[:2], dtype=bool)
    if rgb.shape[-1] == 4:
        mask &= rgb[..., 3] > 0
    pixels = lab[mask]
    if pixels.size == 0:
        pixels = lab.reshape(-1, 3)
        mask = np.ones(lab.shape[:2], dtype=bool)
    return lab, mask, pixels.mean(axis=0), pixels.std(axis=0)


def _pool_stats(stats, window: int):
    if window <= 1 or len(stats) <= 1:
        return stats
    radius = window // 2
    values = np.stack([(item[2], item[3]) for item in stats])
    pooled = []
    for i, item in enumerate(stats):
        mean, std = values[np.arange(i - radius, i + radius + 1) % len(stats)].mean(axis=0)
        pooled.append((item[0], item[1], mean, std))
    return pooled


class MatchColorsToReferenceNode:
    """color_match.py Lab transfer, with optional temporal statistic stabilization."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched_frames",)
    FUNCTION = "match"
    CATEGORY = "color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frame_window": ("INT", {"default": 1, "min": 1, "max": 99, "step": 2}),
                "mask": (["auto", "none"], {"default": "auto"}),
                "bg_threshold": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {"image_ref": ("IMAGE",)},
        }

    def match(self, image_target, strength=1.0, frame_window=1, mask="auto", bg_threshold=18.0, image_ref=None):
        if strength <= 0.0:
            return (image_target,)
        target = (image_target.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        target_stats = [_frame_stats(frame, mask, float(bg_threshold)) for frame in target]
        pooled = _pool_stats(target_stats, int(frame_window))

        reference_stats = None
        if image_ref is not None:
            reference = (image_ref.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            if reference.shape[0] not in (1, target.shape[0]):
                raise ValueError("Reference batch must contain one image or match the target batch size.")
            reference_stats = [_frame_stats(frame, mask, float(bg_threshold)) for frame in reference]

        results = []
        for i, frame in enumerate(target):
            lab, target_mask, own_mean, own_std = target_stats[i]
            if reference_stats is None:
                source_mean, source_std = own_mean, own_std
                destination_mean, destination_std = pooled[i][2], pooled[i][3]
            else:
                source_mean, source_std = pooled[i][2], pooled[i][3]
                ref = reference_stats[0 if len(reference_stats) == 1 else i]
                destination_mean, destination_std = ref[2], ref[3]

            matched = (lab - source_mean) * (destination_std / np.maximum(source_std, EPS)) + destination_mean
            out_lab = np.clip(lab * (1.0 - float(strength)) + matched * float(strength), 0, 255)
            if mask == "auto":
                out_lab[~target_mask] = lab[~target_mask]
            out = cv2.cvtColor(cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
            if frame.shape[-1] == 4:
                out = np.dstack((out, frame[..., 3]))
            results.append(out)
        return (torch.from_numpy(np.stack(results)).float().div(255.0).to(image_target.device).clamp(0.0, 1.0),)
