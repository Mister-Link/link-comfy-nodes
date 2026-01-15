#!/usr/bin/env python3
"""Post-install script to patch Impact Pack for 5D tensor compatibility."""

import os
import sys
from pathlib import Path


def patch_segs_paste():
    """Patch SEGSPaste in Impact Pack to handle 5D tensors."""

    # Find Impact Pack installation
    impact_segs_path = None
    search_paths = [
        Path(
            "/workspace/ComfyUI/custom_nodes/comfyui-impact-pack/modules/impact/segs_nodes.py"
        ),
        Path(
            "/workspace/ComfyUI/custom_nodes/ComfyUI-Impact-Pack/modules/impact/segs_nodes.py"
        ),
        Path(
            "/home/developer/ComfyUI/custom_nodes/comfyui-impact-pack/modules/impact/segs_nodes.py"
        ),
        Path(
            "/home/developer/ComfyUI/custom_nodes/ComfyUI-Impact-Pack/modules/impact/segs_nodes.py"
        ),
    ]

    for path in search_paths:
        if path.exists():
            impact_segs_path = path
            break

    if not impact_segs_path:
        print(
            "[link-comfy-nodes] Warning: Could not find Impact Pack installation, skipping patch"
        )
        return False

    print(f"[link-comfy-nodes] Found Impact Pack at: {impact_segs_path}")

    # Read the file
    with open(impact_segs_path, "r") as f:
        content = f.read()

    # Check if already patched
    if "# Normalize cropped_image if it has extra dimensions" in content:
        print("[link-comfy-nodes] SEGSPaste already patched, skipping")
        return True

    # Apply patch
    old_code = """                        if isinstance(cropped_image, np.ndarray):
                            cropped_image = torch.from_numpy(cropped_image)
                        ref_image = cropped_image[i].unsqueeze(0)"""

    new_code = """                        if isinstance(cropped_image, np.ndarray):
                            cropped_image = torch.from_numpy(cropped_image)
                        # Normalize cropped_image if it has extra dimensions
                        while cropped_image.ndim > 4:
                            squeeze_dim = None
                            for dim in range(cropped_image.ndim):
                                if cropped_image.shape[dim] == 1:
                                    squeeze_dim = dim
                                    break
                            if squeeze_dim is not None:
                                cropped_image = cropped_image.squeeze(squeeze_dim)
                            else:
                                cropped_image = cropped_image[0]
                        ref_image = cropped_image[i].unsqueeze(0)"""

    if old_code not in content:
        print(
            "[link-comfy-nodes] Warning: Could not find target code to patch, Impact Pack may have been updated"
        )
        return False

    # Backup original
    backup_path = impact_segs_path.with_suffix(".py.link-comfy-backup")
    if not backup_path.exists():
        with open(backup_path, "w") as f:
            f.write(content)
        print(f"[link-comfy-nodes] Created backup at: {backup_path}")

    # Apply patch
    patched_content = content.replace(old_code, new_code)

    with open(impact_segs_path, "w") as f:
        f.write(patched_content)

    print("[link-comfy-nodes] Successfully patched SEGSPaste for 5D tensor support")
    return True


if __name__ == "__main__":
    try:
        success = patch_segs_paste()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[link-comfy-nodes] Error during patching: {e}")
        sys.exit(1)
