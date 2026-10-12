import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "nodes" / "pixel_art" / "pixel_effect.py"
SPEC = importlib.util.spec_from_file_location("pixel_effect", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_even_kernel_keeps_selected_and_fallback_grids_aligned():
    """Regression for kernel=8, pixel_size=5 on dimensions congruent to 1 mod 5."""
    model = MODULE.PixelEffectModule().eval()
    rgb = torch.rand(1, 3, 11, 16)
    alpha = torch.full((1, 1, 11, 16), 255.0)

    result_rgb, result_alpha, *_ = model(
        rgb,
        alpha,
        param_num_bins=32,
        param_kernel_size=8,
        param_pixel_size=5,
        alpha_threshold=0.58,
    )

    assert result_rgb.shape == (1, 3, 15, 20)
    assert result_alpha.shape == (1, 1, 15, 20)
