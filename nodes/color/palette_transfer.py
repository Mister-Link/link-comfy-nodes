"""Palette transfer utilities for color grading."""

from __future__ import annotations

import numpy as np


def _srgb_to_linear01(srgb01: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(srgb01 <= 0.04045, srgb01 / 12.92, ((srgb01 + a) / (1 + a)) ** 2.4)


def _linear01_to_srgb(linear01: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        linear01 <= 0.0031308, 12.92 * linear01, (1 + a) * (linear01 ** (1 / 2.4)) - a
    )


def rgb_u8_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 (..., 3) to CIE Lab float32 (..., 3), D65."""
    rgb01 = rgb_u8.astype(np.float32) / 255.0
    rgb_lin = _srgb_to_linear01(rgb01)

    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = rgb_lin @ m.T

    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = xyz / white

    eps = 216 / 24389
    kappa = 24389 / 27

    def f(t):
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    l_val = 116 * fy - 16
    a_val = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return np.stack([l_val, a_val, b_val], axis=-1).astype(np.float32)


def lab_to_rgb_u8(lab: np.ndarray) -> np.ndarray:
    """Convert CIE Lab float (..., 3) to sRGB uint8 (..., 3), D65."""
    l_val = lab[..., 0]
    a_val = lab[..., 1]
    b_val = lab[..., 2]

    fy = (l_val + 16) / 116
    fx = fy + (a_val / 500)
    fz = fy - (b_val / 200)

    eps = 216 / 24389
    kappa = 24389 / 27

    def f_inv(t):
        return np.where(t**3 > eps, t**3, (116 * t - 16) / kappa)

    x = f_inv(fx)
    y = f_inv(fy)
    z = f_inv(fz)

    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = np.stack([x, y, z], axis=-1) * white

    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    )
    rgb_lin = xyz @ m_inv.T
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)
    rgb01 = _linear01_to_srgb(rgb_lin)
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return np.rint(rgb01 * 255.0).astype(np.uint8)


def detect_background_color(image_rgb: np.ndarray) -> np.ndarray:
    """Return the most common color in the image."""
    pixels = image_rgb.reshape(-1, 3)
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    return unique[np.argmax(counts)]


def reinhard_transfer_lab(
    target_rgb_u8: np.ndarray,
    reference_rgb_u8: np.ndarray,
    target_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> np.ndarray:
    """Lab mean/std transfer (Reinhard) on masked pixels only."""
    tgt_lab = rgb_u8_to_lab(target_rgb_u8)
    ref_lab = rgb_u8_to_lab(reference_rgb_u8)

    tgt = tgt_lab[target_mask]
    ref = ref_lab[reference_mask]

    if tgt.size == 0 or ref.size == 0:
        return target_rgb_u8.copy()

    tgt_mean = tgt.mean(axis=0)
    tgt_std = tgt.std(axis=0)
    ref_mean = ref.mean(axis=0)
    ref_std = ref.std(axis=0)

    eps = 1e-6
    scale = ref_std / np.maximum(tgt_std, eps)

    out = tgt_lab.copy()
    out[target_mask] = (out[target_mask] - tgt_mean) * scale + ref_mean

    out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)
    out[..., 1] = np.clip(out[..., 1], -128.0, 127.0)
    out[..., 2] = np.clip(out[..., 2], -128.0, 127.0)
    return lab_to_rgb_u8(out)
