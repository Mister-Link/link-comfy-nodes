#!/usr/bin/env python3
"""Test script for BatchImageSave node"""

import os
import sys

import numpy as np
import torch

# Add ComfyUI to path
sys.path.insert(0, "/home/developer/ComfyUI")
sys.path.insert(0, "/home/developer/ComfyUI/custom_nodes/link-comfy-nodes")

from nodes.video_nodes import BatchImageSave


def test_batch_image_save():
    """Test the BatchImageSave node with sample data"""

    # Create a batch of 5 test images (64x64 RGB)
    num_images = 5
    height, width = 64, 64

    # Create test images with different colors
    images_list = []
    for i in range(num_images):
        # Create a solid color image
        color = [(i * 50) % 255, (i * 100) % 255, (i * 150) % 255]
        img = np.ones((height, width, 3), dtype=np.float32)
        img[:, :, 0] = color[0] / 255.0
        img[:, :, 1] = color[1] / 255.0
        img[:, :, 2] = color[2] / 255.0
        images_list.append(img)

    # Stack into batch tensor
    images_batch = torch.from_numpy(np.stack(images_list))

    print(f"Created test batch: {images_batch.shape}")
    print(f"  - Number of images: {num_images}")
    print(f"  - Image size: {height}x{width}")
    print(f"  - Channels: 3 (RGB)")

    # Test 1: Basic formatting with {:02d}
    print("\n=== Test 1: output/loop_42/frame_{:02d}.png ===")
    node = BatchImageSave()
    result = node.save_images(images_batch, "output/loop_42/frame_{:02d}.png")
    print(f"Result: {result}")

    # Test 2: Different formatting with {:03d}
    print("\n=== Test 2: test/batch/img_{:03d}.png ===")
    result = node.save_images(images_batch, "test/batch/img_{:03d}.png")
    print(f"Result: {result}")

    # Test 3: Simple filename without subdirectory
    print("\n=== Test 3: simple_{:04d}.png ===")
    result = node.save_images(images_batch, "simple_{:04d}.png")
    print(f"Result: {result}")

    print("\n✓ All tests completed successfully!")
    print("\nNote: Files are saved relative to ComfyUI's output directory")


if __name__ == "__main__":
    test_batch_image_save()
