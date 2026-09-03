from __future__ import annotations

import torch

_XN, _YN, _ZN = 0.95047, 1.0, 1.08883
_EPS3 = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0

_M_RGB2XYZ = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
]
_M_XYZ2RGB = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
]


def _rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(_M_RGB2XYZ, device=rgb.device, dtype=rgb.dtype)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).clamp(min=0) ** 2.4)
    xyz = linear @ m.T
    xr, yr, zr = xyz[..., 0] / _XN, xyz[..., 1] / _YN, xyz[..., 2] / _ZN

    def f(t):
        return torch.where(t > _EPS3, t.clamp(min=1e-12) ** (1.0 / 3.0), (_KAPPA * t + 16.0) / 116.0)

    fx, fy, fz = f(xr), f(yr), f(zr)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)


def _lab_to_rgb_torch(lab: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(_M_XYZ2RGB, device=lab.device, dtype=lab.dtype)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def finv(f):
        f3 = f ** 3
        return torch.where(f3 > _EPS3, f3, (116.0 * f - 16.0) / _KAPPA)

    xr, yr, zr = finv(fx), finv(fy), finv(fz)
    xyz = torch.stack([xr * _XN, yr * _YN, zr * _ZN], dim=-1)
    linear = xyz @ m.T
    rgb = torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.clamp(min=0) ** (1 / 2.4) - 0.055)
    return rgb.clamp(0.0, 1.0)


def _balanced_sample_indices(points, max_fit_samples, per_bin_cap, bin_l, bin_ab, generator):
    n = points.shape[0]
    device = points.device
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    ql = torch.floor(points[:, 0] / bin_l).to(torch.int64)
    qa = torch.floor((points[:, 1] + 128.0) / bin_ab).to(torch.int64)
    qb = torch.floor((points[:, 2] + 128.0) / bin_ab).to(torch.int64)
    qa -= qa.min()
    qb -= qb.min()
    ql -= ql.min()
    na = int(qa.max().item()) + 1
    nb = int(qb.max().item()) + 1
    key = ql * (na * nb) + qa * nb + qb

    # random priority breaks ties within a bin so the capped subset is an
    # unbiased random pick, matching the spirit of the original per-bin rng.choice
    priority = torch.rand(n, generator=generator, device=device)
    order1 = torch.argsort(priority)
    order2 = torch.argsort(key[order1], stable=True)
    order = order1[order2]

    sorted_key = key[order]
    _, counts = torch.unique_consecutive(sorted_key, return_counts=True)
    starts = torch.cumsum(counts, dim=0) - counts
    starts_full = torch.repeat_interleave(starts, counts)
    rank = torch.arange(n, device=device) - starts_full
    idx = order[rank < per_bin_cap]

    if idx.shape[0] > max_fit_samples:
        perm = torch.randperm(idx.shape[0], generator=generator, device=device)[:max_fit_samples]
        idx = idx[perm]
    return idx


def _trim_torch(points, trim_low, trim_high):
    if points.shape[0] == 0:
        return points
    lo = torch.quantile(points, trim_low / 100.0, dim=0)
    hi = torch.quantile(points, trim_high / 100.0, dim=0)
    keep = torch.all((points >= lo) & (points <= hi), dim=1)
    trimmed = points[keep]
    if trimmed.shape[0] >= max(100, int(0.30 * points.shape[0])):
        return trimmed
    return points


def _mean_cov_torch(points):
    mean = points.mean(dim=0)
    centered = points - mean
    n = points.shape[0]
    cov = (centered.T @ centered) / max(n - 1, 1)
    cov = cov + torch.eye(3, device=points.device, dtype=points.dtype) * 1e-4
    return mean, cov


def _sqrtm_psd_torch(matrix, inverse=False):
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp(min=1e-8)
    values = (1.0 / values.sqrt()) if inverse else values.sqrt()
    return (vectors * values) @ vectors.T


class MatchColorsToReferenceV2Node:
    """v2 of Match Colors to Reference: fixes an O(bins x pixels) sampler and runs
    entirely on-device in torch (no OpenCV/numpy CPU round-trip)."""

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
        if strength <= 0.0 or image_ref is None:
            return (image_target,)

        device = image_target.device
        dtype = torch.float32
        target = image_target.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        reference = image_ref.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        n = target.shape[0]
        if reference.shape[0] not in (1, n):
            raise ValueError("Reference batch must contain one image or match the target batch size.")

        tgt_lab_all = _rgb_to_lab_torch(target[..., :3])
        ref_shared = reference.shape[0] == 1
        ref_lab_all = _rgb_to_lab_torch(reference[..., :3])

        cached_ref = None
        results = []
        for i in range(n):
            gen_ref = torch.Generator(device=device)
            gen_ref.manual_seed(int(seed) + i * 2)
            gen_tgt = torch.Generator(device=device)
            gen_tgt.manual_seed(int(seed) + i * 2 + 1)

            tgt_lab = tgt_lab_all[i]
            tgt_points = tgt_lab.reshape(-1, 3)

            if ref_shared and cached_ref is not None:
                ref_mean, ref_sqrt_cov = cached_ref
            else:
                ref_lab = ref_lab_all[0 if ref_shared else i]
                ref_points = ref_lab.reshape(-1, 3)
                ref_idx = _balanced_sample_indices(
                    ref_points, int(max_fit_samples), int(per_bin_cap), float(bin_L), float(bin_ab), gen_ref)
                ref_fit = _trim_torch(ref_points[ref_idx], float(trim_low), float(trim_high))
                ref_mean, ref_cov = _mean_cov_torch(ref_fit)
                ref_sqrt_cov = _sqrtm_psd_torch(ref_cov)
                if ref_shared:
                    cached_ref = (ref_mean, ref_sqrt_cov)

            tgt_idx = _balanced_sample_indices(
                tgt_points, int(max_fit_samples), int(per_bin_cap), float(bin_L), float(bin_ab), gen_tgt)
            tgt_fit = _trim_torch(tgt_points[tgt_idx], float(trim_low), float(trim_high))
            tgt_mean, tgt_cov = _mean_cov_torch(tgt_fit)
            tgt_sqrt_inv = _sqrtm_psd_torch(tgt_cov, inverse=True)

            transform = ref_sqrt_cov @ tgt_sqrt_inv
            moved = (tgt_points - tgt_mean) @ transform.T + ref_mean
            moved = torch.stack([
                moved[:, 0].clamp(0.0, 100.0),
                moved[:, 1].clamp(-128.0, 127.0),
                moved[:, 2].clamp(-128.0, 127.0),
            ], dim=-1).reshape(tgt_lab.shape)

            blended_lab = tgt_lab * (1.0 - float(strength)) + moved * float(strength)
            out = _lab_to_rgb_torch(blended_lab)
            if target.shape[-1] == 4:
                out = torch.cat([out, target[i, ..., 3:4]], dim=-1)
            results.append(out)

        return (torch.stack(results, dim=0).clamp(0.0, 1.0).to(image_target.dtype),)
