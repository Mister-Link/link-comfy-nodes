from __future__ import annotations

import cv2
import numpy as np
import torch

EPS = 1e-6


def _foreground_mask(bgr, threshold):
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
    mask = (np.linalg.norm(lab - background, axis=2) > float(threshold)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def _match_one(reference_rgb, target_rgb, strength, mask_mode, threshold):
    ref_bgr = cv2.cvtColor(reference_rgb[..., :3], cv2.COLOR_RGB2BGR)
    tgt_bgr = cv2.cvtColor(target_rgb[..., :3], cv2.COLOR_RGB2BGR)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_mask = _foreground_mask(ref_bgr, threshold) if mask_mode == "auto" else None
    tgt_mask = _foreground_mask(tgt_bgr, threshold) if mask_mode == "auto" else None
    ref_px = ref_lab[ref_mask] if ref_mask is not None else ref_lab.reshape(-1, 3)
    tgt_px = tgt_lab[tgt_mask] if tgt_mask is not None else tgt_lab.reshape(-1, 3)
    if ref_px.size == 0 or tgt_px.size == 0:
        return target_rgb.copy()
    ref_mean, ref_std = ref_px.mean(axis=0), ref_px.std(axis=0)
    tgt_mean, tgt_std = tgt_px.mean(axis=0), tgt_px.std(axis=0)
    matched = (tgt_lab - tgt_mean) * (ref_std / np.maximum(tgt_std, EPS)) + ref_mean
    out_lab = np.clip(tgt_lab * (1.0 - strength) + matched * strength, 0, 255)
    if tgt_mask is not None:
        out_lab[~tgt_mask] = tgt_lab[~tgt_mask]
    out = cv2.cvtColor(cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    return np.dstack((out, target_rgb[..., 3])) if target_rgb.shape[-1] == 4 else out


class MatchColorsImageNode:
    """Single-image Lab color transfer with optional flat-background protection."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched_image",)
    FUNCTION = "match"
    CATEGORY = "color"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_target": ("IMAGE",),
            "image_reference": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "mask": (["auto", "none"], {"default": "auto"}),
            "bg_threshold": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 100.0, "step": 1.0}),
        }}

    def match(self, image_target, image_reference, strength=1.0, mask="auto", bg_threshold=18.0):
        if strength <= 0.0:
            return (image_target,)
        target = (image_target.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        reference = (image_reference.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        if reference.shape[0] not in (1, target.shape[0]):
            raise ValueError("Reference batch must contain one image or match the target batch size.")
        results = [_match_one(reference[0 if reference.shape[0] == 1 else i], frame,
                              float(strength), mask, float(bg_threshold))
                   for i, frame in enumerate(target)]
        return (torch.from_numpy(np.stack(results)).float().div(255.0).to(image_target.device).clamp(0.0, 1.0),)
