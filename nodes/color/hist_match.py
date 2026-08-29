from __future__ import annotations

import cv2
import numpy as np
import torch

EPS = 1e-6

def _rgb_to_lab_float(rgb):
    return cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB).astype(np.float32)

def _lab_to_rgb_float(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0.0, 1.0)

def _balanced_sample(points, max_fit_samples, per_bin_cap, bin_l, bin_ab, seed):
    rng = np.random.default_rng(seed)
    ql = np.floor(points[:, 0] / bin_l).astype(np.int32)
    qa = np.floor((points[:, 1] + 128.0) / bin_ab).astype(np.int32)
    qb = np.floor((points[:, 2] + 128.0) / bin_ab).astype(np.int32)
    _, inverse = np.unique(np.stack([ql, qa, qb], axis=1), axis=0, return_inverse=True)
    collected = []
    for i in range(int(inverse.max()) + 1 if len(inverse) else 0):
        idx = np.flatnonzero(inverse == i)
        if len(idx) > per_bin_cap:
            idx = rng.choice(idx, size=per_bin_cap, replace=False)
        collected.append(idx)
    if not collected:
        return points.copy()
    idx = np.concatenate(collected)
    if len(idx) > max_fit_samples:
        idx = rng.choice(idx, size=max_fit_samples, replace=False)
    return points[idx].copy()

def _trim(points, trim_low, trim_high):
    lo, hi = np.percentile(points, trim_low, axis=0), np.percentile(points, trim_high, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    trimmed = points[keep]
    return trimmed if len(trimmed) >= max(100, int(0.30 * len(points))) else points

def _mean_cov(points):
    mean = points.mean(axis=0)
    centered = points - mean
    cov = (centered.T @ centered) / max(len(points) - 1, 1)
    return mean.astype(np.float32), (cov + np.eye(3, dtype=np.float32) * 1e-4).astype(np.float32)

def _sqrtm_psd(matrix, inverse=False):
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-8, None)
    values = (1.0 / np.sqrt(values)) if inverse else np.sqrt(values)
    return (vectors * values) @ vectors.T

def _transform(reference_rgb, target_rgb, max_fit_samples, per_bin_cap, bin_l, bin_ab,
               trim_low, trim_high, seed):
    ref_lab, tgt_lab = _rgb_to_lab_float(reference_rgb), _rgb_to_lab_float(target_rgb)
    ref_points, tgt_points = ref_lab.reshape(-1, 3), tgt_lab.reshape(-1, 3)
    ref_fit = _trim(_balanced_sample(ref_points, max_fit_samples, per_bin_cap, bin_l, bin_ab, seed), trim_low, trim_high)
    tgt_fit = _trim(_balanced_sample(tgt_points, max_fit_samples, per_bin_cap, bin_l, bin_ab, seed + 1), trim_low, trim_high)
    ref_mean, ref_cov = _mean_cov(ref_fit)
    tgt_mean, tgt_cov = _mean_cov(tgt_fit)
    transform = _sqrtm_psd(ref_cov) @ _sqrtm_psd(tgt_cov, inverse=True)
    moved = (tgt_points - tgt_mean) @ transform.T + ref_mean
    moved[:, 0] = np.clip(moved[:, 0], 0.0, 100.0)
    moved[:, 1] = np.clip(moved[:, 1], -128.0, 127.0)
    moved[:, 2] = np.clip(moved[:, 2], -128.0, 127.0)
    return tgt_lab, moved.reshape(tgt_lab.shape)

class MatchColorsToReferenceNode:
    """Balanced full-covariance Lab color transfer based on color_match_v6c_balanced_cov.py."""
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched_frames",)
    FUNCTION = "match"
    CATEGORY = "color"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_target": ("IMAGE",),
            "strength": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01}),
            "max_fit_samples": ("INT", {"default": 60000, "min": 1000, "max": 500000, "step": 1000}),
            "per_bin_cap": ("INT", {"default": 300, "min": 1, "max": 10000, "step": 10}),
            "bin_L": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 50.0, "step": 0.5}),
            "bin_ab": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 50.0, "step": 0.5}),
            "trim_low": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 49.0, "step": 0.1}),
            "trim_high": ("FLOAT", {"default": 99.5, "min": 51.0, "max": 100.0, "step": 0.1}),
            "seed": ("INT", {"default": 7, "min": 0, "max": 0x7fffffff}),
        }, "optional": {"image_ref": ("IMAGE",)}}

    def match(self, image_target, strength=0.88, max_fit_samples=60000,
              per_bin_cap=300, bin_L=4.0, bin_ab=6.0, trim_low=0.5, trim_high=99.5, seed=7, image_ref=None):
        if strength <= 0.0:
            return (image_target,)
        target = image_target.detach().cpu().numpy().clip(0.0, 1.0)
        if image_ref is None:
            return (image_target,)
        reference = image_ref.detach().cpu().numpy().clip(0.0, 1.0)
        if reference.shape[0] not in (1, target.shape[0]):
            raise ValueError("Reference batch must contain one image or match the target batch size.")
        results = []
        for i, frame in enumerate(target):
            ref = reference[0 if reference.shape[0] == 1 else i]
            target_lab, moved_lab = _transform(ref[..., :3], frame[..., :3], int(max_fit_samples), int(per_bin_cap), float(bin_L), float(bin_ab), float(trim_low), float(trim_high), int(seed) + i * 2)
            out = _lab_to_rgb_float(target_lab * (1.0 - float(strength)) + moved_lab * float(strength))
            if frame.shape[-1] == 4:
                out = np.dstack((out, frame[..., 3]))
            results.append(out)
        return (torch.from_numpy(np.stack(results)).float().to(image_target.device).clamp(0.0, 1.0),)
