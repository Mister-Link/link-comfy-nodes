#!/usr/bin/env python
"""Compare temporal flicker of ConvertToPixelArt with and without stabilization.

Runs the node on a video, measures frame-to-frame output change restricted to
regions where the *input* is static (i.e. change there is flicker, not motion),
and writes side-by-side result videos.

Usage (from any venv with torch + ffmpeg on PATH):
    python test_stabilization.py [--video ~/Downloads/before.mp4] [--outdir ~/Downloads]
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

NODE_DIR = Path(__file__).resolve().parent / "nodes" / "pixel_art"


def load_node_class():
    """Import nodes/pixel_art without triggering the repo's full __init__."""
    pkg = types.ModuleType("pixel_art")
    pkg.__path__ = [str(NODE_DIR)]
    sys.modules["pixel_art"] = pkg
    for mod_name in ("pixel_effect", "node"):
        spec = importlib.util.spec_from_file_location(
            f"pixel_art.{mod_name}", NODE_DIR / f"{mod_name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"pixel_art.{mod_name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["pixel_art.node"].ConvertToPixelArt


def read_video(path: Path):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    w, h, fps = probe.stdout.strip().split(",")
    w, h = int(w), int(h)
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    return frames, fps


def write_video(path: Path, frames: np.ndarray, fps: str):
    # yuv420p needs even dimensions; crop a row/col if necessary
    frames = frames[:, : frames.shape[1] // 2 * 2, : frames.shape[2] // 2 * 2]
    n, h, w, _ = frames.shape
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
            "-r", fps, "-i", "-",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path),
        ],
        input=frames.tobytes(), check=True,
    )


def flicker_metric(inp: torch.Tensor, out: torch.Tensor) -> float:
    """Mean |output delta| (0-255 scale) over pixels where the input is static.

    Output is a different resolution than input; both deltas are compared on
    the output grid by nearest-resizing the input-motion mask.
    """
    oh, ow = out.shape[1], out.shape[2]
    in_delta = (inp[1:] - inp[:-1]).abs().mean(dim=-1)  # (N-1, H, W)
    static = (in_delta < 4.0 / 255.0).float().unsqueeze(1)  # (N-1, 1, H, W)
    static = torch.nn.functional.interpolate(static, size=(oh, ow), mode="nearest")
    static = static.squeeze(1)
    out_delta = (out[1:] - out[:-1]).abs().mean(dim=-1) * 255.0
    return float((out_delta * static).sum() / static.sum().clamp(min=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(Path.home() / "Downloads" / "before.mp4"))
    ap.add_argument(
        "--mask",
        default=None,
        help="Optional alpha-mask video (white=opaque), e.g. chroma-key output",
    )
    ap.add_argument("--outdir", default=str(Path.home() / "Downloads"))
    ap.add_argument("--kernel-size", type=int, default=9)
    ap.add_argument("--pixel-size", type=int, default=11)
    ap.add_argument("--num-bins", type=int, default=10)
    # 0.0 = raw coverage, no bias; alpha is handled upstream (chroma key)
    # in this workflow. Irrelevant anyway unless --mask is supplied: with
    # no alpha input the node synthesizes full opacity and the threshold
    # is a no-op.
    ap.add_argument("--alpha-threshold", type=float, default=0.0)
    args = ap.parse_args()

    node_cls = load_node_class()
    node = node_cls()

    frames_np, fps = read_video(Path(args.video))
    frames = torch.from_numpy(frames_np.astype(np.float32) / 255.0)
    print(f"{args.video}: {frames.shape[0]} frames {frames.shape[2]}x{frames.shape[1]} @ {fps}")

    if args.mask:
        # Alpha is now embedded-only: join the mask as a 4th channel.
        mask_np, _ = read_video(Path(args.mask))
        mask = torch.from_numpy(mask_np.astype(np.float32) / 255.0).mean(dim=-1)
        if mask.shape != frames.shape[:3]:
            raise SystemExit(
                f"mask shape {tuple(mask.shape)} != video shape {tuple(frames.shape[:3])}"
            )
        frames = torch.cat([frames, mask.unsqueeze(-1)], dim=-1)

    runs = {
        "baseline": dict(stability=0.0),
        "stable_0.5": dict(stability=0.5),
        "stable_1.0": dict(stability=1.0),
    }
    outdir = Path(args.outdir).expanduser()
    common = dict(
        kernel_size=args.kernel_size,
        pixel_size=args.pixel_size,
        num_bins=args.num_bins,
        alpha_threshold=args.alpha_threshold,
    )
    for name, kw in runs.items():
        (out,) = node.convert(frames, **common, **kw)
        m = flicker_metric(frames[..., :3], out[..., :3])
        dest = outdir / f"pixelated_{name}.mp4"
        write_video(dest, (out[..., :3].numpy() * 255.0).round().astype(np.uint8), fps)
        print(f"{name:12s} flicker(static regions) = {m:6.3f}  -> {dest}")


if __name__ == "__main__":
    main()
