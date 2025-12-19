import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from models.module_pixel_effect import PixelEffectModule
from utils import img_common_util


class BatchImageProcessor:
    """
    Process all images in input folder with pixelation effect while preserving alpha channels.
    Uses alpha-aware pixelation for proper transparency handling.
    """

    def __init__(
        self,
        input_dir="input",
        output_dir="output",
        param_num_bins=4,
        param_kernel_size=8,
        param_pixel_size=8,
        alpha_threshold=0.5,
    ):
        """
        Initialize the batch processor.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            param_num_bins: Number of intensity bins for color quantization (default: 4)
            param_kernel_size: Kernel size for pixelation (default: 10)
            param_pixel_size: Pixel size for pixelation (default: 16)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.param_num_bins = param_num_bins
        self.param_kernel_size = param_kernel_size
        self.param_pixel_size = param_pixel_size
        self.alpha_threshold = alpha_threshold

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load model (PixelEffectModule - alpha aware)
        self.model = PixelEffectModule()
        self.model.eval()
        print(
            f"Model loaded. Processing parameters: num_bins={param_num_bins}, "
            f"kernel_size={param_kernel_size}, pixel_size={param_pixel_size}"
        )

    def process_image(self, input_path, output_path):
        """
        Process a single image while preserving alpha channel.

        Args:
            input_path: Path to input image
            output_path: Path to save output image
        """
        try:
            # Load image
            img = Image.open(input_path).convert("RGBA")
            img_np = np.array(img).astype(np.float32)

            # Separate RGB and alpha
            img_rgb = img_np[:, :, :3]
            alpha_channel = img_np[:, :, 3]

            # Ensure dimensions match
            if img_rgb.shape[:2] != alpha_channel.shape[:2]:
                alpha_channel = cv2.resize(
                    alpha_channel,
                    (img_rgb.shape[1], img_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Convert to tensors
            img_pt = torch.from_numpy(
                np.transpose(img_rgb, axes=[2, 0, 1])[np.newaxis, :, :, :]
            )
            alpha_pt = torch.from_numpy(alpha_channel).unsqueeze(0).unsqueeze(0)

            # Process with model (now returns both RGB and alpha)
            with torch.no_grad():
                result_rgb_pt, result_alpha_pt = self.model(
                    img_pt,
                    alpha_pt,
                    param_num_bins=self.param_num_bins,
                    param_kernel_size=self.param_kernel_size,
                    param_pixel_size=self.param_pixel_size,
                    alpha_threshold=self.alpha_threshold,
                )

            # Convert back to numpy
            result_rgb_np = (
                result_rgb_pt[0, ...].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            )
            result_alpha_np = result_alpha_pt[0, 0, :, :].cpu().numpy().astype(np.uint8)

            # Ensure alpha matches RGB size
            if result_rgb_np.shape[:2] != result_alpha_np.shape[:2]:
                result_alpha_np = cv2.resize(
                    result_alpha_np,
                    (result_rgb_np.shape[1], result_rgb_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Combine into RGBA
            result_rgba_np = np.concatenate(
                (result_rgb_np, result_alpha_np[:, :, np.newaxis]), axis=2
            )

            # Save output
            Image.fromarray(result_rgba_np, "RGBA").save(output_path)
            return True, None

        except Exception as e:
            return False, str(e)

    def process_batch(self, image_extensions=None):
        """
        Process all images in input directory.

        Args:
            image_extensions: List of file extensions to process (default: ['.png', '.jpg', '.jpeg', '.bmp', '.gif'])
        """
        if image_extensions is None:
            image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]

        # Get all image files
        input_path = Path(self.input_dir)
        if not input_path.exists():
            print(f"Error: Input directory '{self.input_dir}' does not exist")
            return

        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        image_files = sorted(list(set(image_files)))  # Remove duplicates and sort

        if not image_files:
            print(f"No image files found in '{self.input_dir}'")
            return

        print(f"Found {len(image_files)} images to process")
        print("-" * 60)

        processed = 0
        failed = 0

        for i, input_file in enumerate(image_files, 1):
            filename = input_file.name
            output_file = Path(self.output_dir) / filename

            success, error = self.process_image(str(input_file), str(output_file))

            if success:
                print(f"[{i}/{len(image_files)}] ✓ {filename} -> {output_file.name}")
                processed += 1
            else:
                print(f"[{i}/{len(image_files)}] ✗ {filename} (Error: {error})")
                failed += 1

        print("-" * 60)
        print(f"Processing complete: {processed} succeeded, {failed} failed")
        print(f"Output saved to: {os.path.abspath(self.output_dir)}")


def main():
    """Main entry point for batch processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch process images with pixelation effect while preserving alpha channels"
    )
    parser.add_argument(
        "--input", default="input", help="Input directory (default: input)"
    )
    parser.add_argument(
        "--output", default="output", help="Output directory (default: output)"
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=4,
        help="Number of intensity bins for color quantization (default: 4)",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=12,
        help="Kernel size for pixelation (default: 10)",
    )
    parser.add_argument(
        "--pixel-size",
        type=int,
        default=8,
        help="Pixel size for pixelation (default: 16)",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.5,
        help="Alpha threshold for opaque pixels: >threshold = opaque, else transparent (default: 0.5)",
    )

    args = parser.parse_args()

    processor = BatchImageProcessor(
        input_dir=args.input,
        output_dir=args.output,
        param_num_bins=args.num_bins,
        param_kernel_size=args.kernel_size,
        param_pixel_size=args.pixel_size,
        alpha_threshold=args.alpha_threshold,
    )

    processor.process_batch()


if __name__ == "__main__":
    main()
