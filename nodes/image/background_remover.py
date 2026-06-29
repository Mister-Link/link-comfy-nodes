from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_model: torch.nn.Module | None = None
_device: torch.device | None = None


def _load_model() -> tuple[torch.nn.Module, torch.device]:
    global _model, _device
    if _model is not None:
        return _model, _device

    from huggingface_hub import hf_hub_download

    print("RemoveBackground: downloading BEN2 model files...")
    script_path  = hf_hub_download("PramaLLC/BEN2", "BEN2.py")
    weights_path = hf_hub_download("PramaLLC/BEN2", "BEN2_Base.pth")

    # Import BEN2 module from its cached path
    spec = importlib.util.spec_from_file_location("BEN2", script_path)
    ben2_module = importlib.util.module_from_spec(spec)
    sys.modules["BEN2"] = ben2_module
    spec.loader.exec_module(ben2_module)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RemoveBackground: loading BEN2 on {device}")

    model = ben2_module.BEN_Base().to(device).eval()
    model.loadcheckpoints(weights_path)

    _model, _device = model, device
    print("RemoveBackground: BEN2 ready")
    return _model, _device


_transform_fp16 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float16),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

_transform_fp32 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


def _infer(frame: np.ndarray, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """frame: float32 [H, W, C] in [0,1] → float32 [H, W] alpha in [0,1]"""
    pil = Image.fromarray((frame[..., :3] * 255).clip(0, 255).astype(np.uint8))
    orig_w, orig_h = pil.size  # PIL: (width, height)
    resized = pil.resize((1024, 1024), Image.LANCZOS)

    transform = _transform_fp16 if device.type == "cuda" else _transform_fp32
    inp = transform(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        result = model(inp)  # [1, 1, 1024, 1024]

    # Resize to original, normalize to [0, 1]
    result = F.interpolate(result.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    lo, hi = result.min(), result.max()
    result = (result - lo) / (hi - lo).clamp(min=1e-6)
    alpha = result.squeeze().cpu().numpy()  # [H, W] in [0, 1]

    # Port the JS contrast-boost that removes white fringing:
    #   normalized = (raw − min) / range × 255
    #   boosted    = clamp((normalized − 8) × 1.12, 0, 255)
    #   alpha      = 0 if boosted < 12 else boosted
    # Note: raw is already normalized above so this simplifies to a contrast stretch.
    a255 = alpha * 255.0
    boosted = np.clip((a255 - 8.0) * 1.12, 0.0, 255.0)
    boosted[boosted < 12.0] = 0.0
    return (boosted / 255.0).astype(np.float32)


class PixPunkRemoveBackground:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "remove_background"
    CATEGORY = "image/transform"
    DESCRIPTION = "Removes image backgrounds locally using BEN2 (GPU accelerated). No data leaves your machine."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def remove_background(
        self,
        images: torch.Tensor,
        unique_id: str | None = None,
    ):
        frames = images.detach().cpu().float()
        if frames.ndim != 4:
            raise ValueError("Expected images with shape (N, H, W, C)")

        model, device = _load_model()

        total = frames.shape[0]
        progress_bar = ProgressBar(total, node_id=unique_id) if ProgressBar else None

        results: list[np.ndarray] = []

        for idx, frame in enumerate(frames.numpy()):
            alpha = _infer(frame, model, device)
            rgba = np.concatenate([frame[..., :3], alpha[..., np.newaxis]], axis=-1)
            results.append(rgba)

            print(f"RemoveBackground: [{idx + 1}/{total}] done")
            if progress_bar is not None:
                progress_bar.update(1)

        return (torch.from_numpy(np.stack(results)),)
