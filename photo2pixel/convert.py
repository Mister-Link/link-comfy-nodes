import os

import cv2
import numpy as np
import torch
from models.module_pixel_effect import PixelEffectModule
from PIL import Image
from utils import img_common_util


def convert_video(input_video="test.mp4", output_video="result.mp4", output_fps=20):
    """
    Process video frames with pixelization effect.

    Args:
        input_video: Path to input MP4 file
        output_video: Path to output MP4 file
        output_fps: Frames per second for output video
    """
    output_dir = "frames"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Processing video: {frame_count} frames at {fps} FPS, {width}x{height}")

    # Load pixelation model
    pixel_model = PixelEffectModule()
    pixel_model.eval()
    print("Model loaded. Starting processing...")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, output_fps, (width, height))

    frame_num = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            img_input = Image.fromarray(frame_rgb)

            # Convert to tensor for pixelation
            img_pt_input = img_common_util.convert_image_to_tensor(
                img_input, preserve_alpha=False
            )

            # Process frame with pixel effect
            img_pt_output = pixel_model(
                img_pt_input, param_num_bins=4, param_kernel_size=6, param_pixel_size=6
            )

            # Convert back to image (RGB)
            img_output = img_common_util.convert_tensor_to_image(
                img_pt_output, has_alpha=False
            )

            # Save individual frame
            output_path = os.path.join(output_dir, f"frame_{frame_num:06d}.png")
            img_output.save(output_path)

            # Resize to original dimensions if needed and write to video
            img_output_resized = img_output.resize((width, height))
            frame_array = np.array(img_output_resized)

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            if (frame_num + 1) % 10 == 0:
                print(f"Processed {frame_num + 1}/{frame_count} frames")

            frame_num += 1

    cap.release()
    out.release()
    print(
        f"Conversion complete: {frame_num} frames saved to {output_dir}/ and {output_video}"
    )


if __name__ == "__main__":
    convert_video()
