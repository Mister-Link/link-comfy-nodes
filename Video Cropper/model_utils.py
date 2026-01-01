"""Utilities for loading AnimeSegmentation models."""

import os


def get_model_cache_path(folder_paths_module):
    """Get ComfyUI models folder path for anime-seg checkpoint."""
    models_dir = folder_paths_module.models_dir
    checkpoint_dir = os.path.join(models_dir, "anime_seg")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "anime-seg-isnet.pth")
    return checkpoint_path, checkpoint_dir


def load_anime_seg_model(folder_paths_module, device="cpu"):
    """
    Load AnimeSegmentation model with fallback strategy.

    Fallback order:
    1. Try HuggingFace from_pretrained() (requires internet)
    2. Try ComfyUI models folder checkpoint
    3. Raise helpful error

    Args:
        folder_paths_module: ComfyUI's folder_paths module
        device: Device to load model on ("cpu" or "cuda")

    Returns:
        Loaded AnimeSegmentation model
    """
    from train import AnimeSegmentation

    checkpoint_path, checkpoint_dir = get_model_cache_path(folder_paths_module)

    # Tier 1: Try HuggingFace
    try:
        print(f"[AutoCropper] Loading from HuggingFace on {device}...")
        model = AnimeSegmentation.from_pretrained("skytnt/anime-seg")
        model.to(device).eval()
        print("[AutoCropper] ✓ Loaded from HuggingFace")
        return model
    except Exception as hf_error:
        print(f"[AutoCropper] HuggingFace failed: {hf_error}")

    # Tier 2: Try local checkpoint
    if os.path.exists(checkpoint_path):
        try:
            print(f"[AutoCropper] Loading from {checkpoint_path}...")
            model = AnimeSegmentation.try_load(
                net_name="isnet_is",
                ckpt_path=checkpoint_path,
                map_location=device,
                img_size=640,
            )
            model.to(device).eval()
            print("[AutoCropper] ✓ Loaded from local checkpoint")
            return model
        except Exception as local_error:
            print(f"[AutoCropper] Local load failed: {local_error}")

    # Tier 3: Error with instructions
    raise RuntimeError(
        f"AnimeSegmentation model not found. Tried:\n"
        f"1. HuggingFace download (requires internet)\n"
        f"2. Local checkpoint at {checkpoint_path}\n\n"
        f"To fix:\n"
        f"- Enable internet for automatic download, OR\n"
        f"- Download to: {checkpoint_path}\n"
        f"  From: https://huggingface.co/skytnt/anime-seg/resolve/main/model.safetensors"
    )
