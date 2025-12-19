#!/usr/bin/env python3
"""
Anime frame cropper using skytnt/anime-seg
Inference-only
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from train import AnimeSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AnimeSegmentation.from_pretrained("skytnt/anime-seg")
model.to(device).eval()




# -----------------------------------------------------------------------------
# Segmentation
# -----------------------------------------------------------------------------

@torch.no_grad()
def segment_frame(model, frame, device):
    if frame.shape[2] == 4:
        bgr = frame[:, :, :3]
    else:
        bgr = frame

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device)

    pred = model(tensor)[0, 0].cpu().numpy()
    mask = (pred > 0.5).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255

    return mask


def mask_to_bbox(mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h


def resize_with_padding(img, target_size):
    """
    Resize image to fit within target_size while preserving aspect ratio.
    Pads with transparent pixels (if alpha) or black (if no alpha).
    """
    th, tw = target_size[1], target_size[0]
    h, w = img.shape[:2]

    scale = min(tw / w, th / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create canvas
    if img.shape[2] == 4:
        canvas = np.zeros((th, tw, 4), dtype=img.dtype)
    else:
        canvas = np.zeros((th, tw, 3), dtype=img.dtype)

    # Center placement
    x0 = (tw - new_w) // 2
    y0 = (th - new_h) // 2

    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--padding", type=int, default=0)
    parser.add_argument("-s", "--size", help="WxH e.g. 979x1562")
    args = parser.parse_args()

    target_size = None
    if args.size:
        w, h = args.size.lower().split("x")
        target_size = (int(w), int(h))

    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.webp"))
    if not files:
        raise RuntimeError("No frames found in ./input")

    print(f"Analyzing {len(files)} frames...")
    global_box = None

    for i, f in enumerate(files, 1):
        frame = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        mask = segment_frame(model, frame, device)
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue

        if global_box is None:
            global_box = list(bbox)
        else:
            global_box[0] = min(global_box[0], bbox[0])
            global_box[1] = min(global_box[1], bbox[1])
            global_box[2] = max(global_box[2], bbox[2])
            global_box[3] = max(global_box[3], bbox[3])

        if i % 25 == 0:
            print(f"  [{i}/{len(files)}]")

    if global_box is None:
        raise RuntimeError("No character detected in any frame")

    x1, y1, x2, y2 = global_box

    sample = cv2.imread(str(files[0]))
    H, W = sample.shape[:2]

    x1 = max(0, x1 - args.padding)
    y1 = max(0, y1 - args.padding)
    x2 = min(W, x2 + args.padding)
    y2 = min(H, y2 + args.padding)

    print(f"\nCrop region: ({x1}, {y1}) → ({x2}, {y2})")
    print(f"Cropped size: {x2-x1}x{y2-y1}")

    print("\nCropping frames...")
    for i, f in enumerate(files, 1):
        frame = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        cropped = frame[y1:y2, x1:x2]

        if target_size:
            cropped = resize_with_padding(cropped, target_size)

        cv2.imwrite(str(output_dir / f.name), cropped)

        if i % 25 == 0:
            print(f"  [{i}/{len(files)}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
