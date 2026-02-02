#!/usr/bin/env python3
"""Improved patch for SEGSPaste to properly handle 5D tensors from AnimateDiff detailers."""

from pathlib import Path


def apply_improved_patch():
    """Apply an improved patch to SEGSPaste that properly handles 5D tensors."""

    impact_segs_path = Path(
        "/home/developer/ComfyUI/custom_nodes/ComfyUI-Impact-Pack/modules/impact/segs_nodes.py"
    )

    if not impact_segs_path.exists():
        print(f"[Patch] Could not find {impact_segs_path}")
        return False

    with open(impact_segs_path, "r") as f:
        content = f.read()

    # Check if the improved patch is already applied
    if "# IMPROVED 5D TENSOR NORMALIZATION" in content:
        print("[Patch] Improved patch already applied")
        return True

    # Find the current patched code
    old_patched_code = """                    # ref_image handling
                    if ref_image_opt is None and seg.cropped_image is not None:
                        cropped_image = seg.cropped_image
                        if isinstance(cropped_image, np.ndarray):
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

    new_improved_code = """                    # ref_image handling
                    if ref_image_opt is None and seg.cropped_image is not None:
                        cropped_image = seg.cropped_image
                        if isinstance(cropped_image, np.ndarray):
                            cropped_image = torch.from_numpy(cropped_image)
                        # IMPROVED 5D TENSOR NORMALIZATION
                        # For AnimateDiff detailers, cropped_image may be 5D: [batch, frames, H, W, C]
                        # We need to flatten it to 4D: [batch*frames, H, W, C] for proper indexing
                        while cropped_image.ndim > 4:
                            squeeze_dim = None
                            for dim in range(cropped_image.ndim):
                                if cropped_image.shape[dim] == 1:
                                    squeeze_dim = dim
                                    break
                            if squeeze_dim is not None:
                                cropped_image = cropped_image.squeeze(squeeze_dim)
                            else:
                                # For 5D without singleton dims, assume [batch, frames, H, W, C]
                                # Reshape to [batch*frames, H, W, C]
                                if cropped_image.ndim == 5:
                                    b, f, h, w, c = cropped_image.shape
                                    cropped_image = cropped_image.reshape(b * f, h, w, c)
                                else:
                                    # Fallback for other cases
                                    cropped_image = cropped_image[0]
                        # Safely index and ensure result is 4D
                        if i < len(cropped_image):
                            ref_image = cropped_image[i].unsqueeze(0)
                        else:
                            # Edge case: if index out of bounds, use last frame
                            ref_image = cropped_image[-1].unsqueeze(0)"""

    if old_patched_code not in content:
        print(
            "[Patch] Could not find existing patch code. The file may have been modified."
        )
        print("[Patch] Trying to find unpatched code...")

        # Try to find the original unpatched code
        original_code = """                    # ref_image handling
                    if ref_image_opt is None and seg.cropped_image is not None:
                        cropped_image = seg.cropped_image
                        if isinstance(cropped_image, np.ndarray):
                            cropped_image = torch.from_numpy(cropped_image)
                        ref_image = cropped_image[i].unsqueeze(0)"""

        if original_code in content:
            print("[Patch] Found original unpatched code, applying improved patch...")
            patched_content = content.replace(original_code, new_improved_code)
        else:
            print("[Patch] ERROR: Could not find code to patch!")
            return False
    else:
        print("[Patch] Replacing existing patch with improved version...")
        patched_content = content.replace(old_patched_code, new_improved_code)

    # Backup
    backup_path = impact_segs_path.with_suffix(".py.backup_before_improved")
    if not backup_path.exists():
        with open(backup_path, "w") as f:
            f.write(content)
        print(f"[Patch] Created backup at: {backup_path}")

    # Apply patch
    with open(impact_segs_path, "w") as f:
        f.write(patched_content)

    print("[Patch] ✓ Successfully applied improved 5D tensor patch to SEGSPaste")
    return True


if __name__ == "__main__":
    import sys

    try:
        success = apply_improved_patch()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[Patch] Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
