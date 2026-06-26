from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime", "-q"])
    import onnxruntime as ort

try:
    from comfy.utils import ProgressBar  # type: ignore[import-not-found]
except Exception:
    ProgressBar = None

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "bgeraser"

MODELS: dict[str, dict] = {
    "u2netp – Fast (4 MB)": {
        "url": "https://huggingface.co/robertwt7/bg-remover-models/resolve/main/onnx/u2netp.onnx",
        "filename": "u2netp.onnx",
        "dims": (320, 320),
    },
    "silueta – Balanced (43 MB)": {
        "url": "https://huggingface.co/robertwt7/bg-remover-models/resolve/main/onnx/silueta.onnx",
        "filename": "silueta.onnx",
        "dims": (320, 320),
    },
    "RMBG-1.4 Quantized – Quality (88 MB)": {
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx",
        "filename": "rmbg14_quantized.onnx",
        "dims": (1024, 1024),
    },
    "RMBG-1.4 FP16 – Quality (88 MB)": {
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_fp16.onnx",
        "filename": "rmbg14_fp16.onnx",
        "dims": (1024, 1024),
    },
    "RMBG-1.4 Float32 – Max Quality (176 MB)": {
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx",
        "filename": "rmbg14_fp32.onnx",
        "dims": (1024, 1024),
    },
}

_sessions: dict[str, ort.InferenceSession] = {}


def _download_model(filename: str, url: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    if path.exists():
        return path
    print(f"BgRemover: downloading {filename} from {url} ...")
    tmp = path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"BgRemover: saved {filename} ({path.stat().st_size // (1024*1024)} MB)")
    return path


def _get_session(model_key: str) -> ort.InferenceSession:
    if model_key in _sessions:
        return _sessions[model_key]
    cfg = MODELS[model_key]
    model_path = _download_model(cfg["filename"], cfg["url"])
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    print(f"BgRemover: loading {cfg['filename']} with {providers[0]}")
    session = ort.InferenceSession(str(model_path), providers=providers)
    _sessions[model_key] = session
    return session


def _preprocess(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """image: float32 [H, W, C] in [0,1] → float32 [1, 3, h, w] in [-1, 1]"""
    pil = Image.fromarray((image * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
    pil = pil.resize((w, h), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5                        # normalize to [-1, 1]
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, H, W]
    return arr


def _postprocess(mask_output: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """Squeeze model output to [H, W] float32 in [0, 1], resize to original dims."""
    mask = mask_output.squeeze()
    # sigmoid if values are outside [0, 1] (some models skip it)
    if mask.min() < 0 or mask.max() > 1:
        mask = 1.0 / (1.0 + np.exp(-mask))
    mask = (mask * 255).clip(0, 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask, mode="L").resize((orig_w, orig_h), Image.BILINEAR)
    return np.array(mask_pil, dtype=np.float32) / 255.0


class LocalBackgroundRemoverNode:
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "alpha")
    FUNCTION = "remove_background"
    CATEGORY = "image/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (list(MODELS.keys()),),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def remove_background(
        self,
        images: torch.Tensor,
        model: str,
        unique_id: str | None = None,
    ):
        frames = images.detach().cpu().float()
        if frames.ndim != 4:
            raise ValueError("Expected images with shape (N, H, W, C)")

        cfg = MODELS[model]
        target_h, target_w = cfg["dims"]
        session = _get_session(model)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        total = frames.shape[0]
        progress_bar = ProgressBar(total, node_id=unique_id) if ProgressBar else None

        results_rgb: list[np.ndarray] = []
        results_alpha: list[np.ndarray] = []

        for idx, frame in enumerate(frames.numpy()):
            orig_h, orig_w = frame.shape[:2]
            tensor = _preprocess(frame, target_h, target_w)
            raw_mask = session.run([output_name], {input_name: tensor})[0]
            alpha = _postprocess(raw_mask, orig_h, orig_w)

            rgb = frame[..., :3]
            results_rgb.append(rgb)
            results_alpha.append(alpha)

            print(f"BgRemover: [{idx + 1}/{total}] done")
            if progress_bar is not None:
                progress_bar.update(1)

        images_out = torch.from_numpy(np.stack(results_rgb))
        alpha_out = torch.from_numpy(np.stack(results_alpha))
        return (images_out, alpha_out)
