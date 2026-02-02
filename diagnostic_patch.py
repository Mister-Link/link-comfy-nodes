"""Diagnostic patch to understand the 5D tensor issue in DetailerForEachPipeForAnimateDiff"""

import sys
from pathlib import Path

# Find and patch the animatediff_nodes.py to add debug logging
impact_path = Path(
    "/home/developer/ComfyUI/custom_nodes/ComfyUI-Impact-Pack/modules/impact/animatediff_nodes.py"
)

if impact_path.exists():
    with open(impact_path, "r") as f:
        content = f.read()

    # Check if already patched
    if "# DIAGNOSTIC:" not in content:
        # Add logging after line where new_cropped_image is created
        old_code = """            if enhanced_image_tensor is None:
                new_cropped_image = cropped_image_frames
            else:
                new_cropped_image = enhanced_image_tensor.cpu().numpy()

            new_seg = SEG(new_cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)"""

        new_code = """            if enhanced_image_tensor is None:
                new_cropped_image = cropped_image_frames
            else:
                new_cropped_image = enhanced_image_tensor.cpu().numpy()

            # DIAGNOSTIC: Log tensor shapes
            import logging
            logging.warning(f"[DIAGNOSTIC] new_cropped_image shape: {new_cropped_image.shape}, type: {type(new_cropped_image)}")
            logging.warning(f"[DIAGNOSTIC] enhanced_image_tensor shape: {enhanced_image_tensor.shape if enhanced_image_tensor is not None else 'None'}")

            new_seg = SEG(new_cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)"""

        if old_code in content:
            patched = content.replace(old_code, new_code)

            # Backup
            backup = impact_path.with_suffix(".py.diagnostic_backup")
            if not backup.exists():
                with open(backup, "w") as f:
                    f.write(content)

            with open(impact_path, "w") as f:
                f.write(patched)

            print("[DIAGNOSTIC] Patch applied to animatediff_nodes.py")
            print(
                "Run your workflow again and check the console output for shape information"
            )
        else:
            print("[DIAGNOSTIC] Could not find target code - file may have changed")
    else:
        print("[DIAGNOSTIC] Already patched")
else:
    print(f"[DIAGNOSTIC] Could not find {impact_path}")
