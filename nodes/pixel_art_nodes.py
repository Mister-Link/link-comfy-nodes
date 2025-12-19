import torch
import numpy as np
from PIL import Image

from ..photo2pixel.module_pixel_effect import PixelEffectModule
from ..photo2pixel import img_common_util


class ConvertToPixelArt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "kernel_size": ("INT", {"default": 9, "min": 1, "max": 100}),
                "pixel_size": ("INT", {"default": 11, "min": 1, "max": 100}),
                "num_bins": ("INT", {"default": 10, "min": 1, "max": 30}),
                "alpha_threshold": (
                    "FLOAT",
                    {"default": 0.58, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "convert"

    CATEGORY = "pixelart"

    def convert(self, frames, kernel_size, pixel_size, num_bins, alpha_threshold):
        pixel_model = PixelEffectModule()
        pixel_model.eval()

        output_frames = []
        output_masks = []

        for frame in frames:
            # Convert torch tensor to PIL Image
            img_np = (frame.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, "RGB")

            # Create a full alpha channel, as input is RGB
            alpha_pil = Image.new("L", img_pil.size, 255)

            # Convert image and alpha to tensors for the model
            img_pt = img_common_util.convert_image_to_tensor(
                img_pil, preserve_alpha=False
            )

            alpha_np = np.array(alpha_pil).astype(np.float32)
            alpha_pt = torch.from_numpy(alpha_np[np.newaxis, np.newaxis, :, :])

            with torch.no_grad():
                result_rgb_pt, result_alpha_pt = pixel_model(
                    img_pt,
                    alpha=alpha_pt,
                    param_num_bins=num_bins,
                    param_kernel_size=kernel_size,
                    param_pixel_size=pixel_size,
                    alpha_threshold=alpha_threshold,
                )

            # Process RGB output
            result_rgb_pt = result_rgb_pt.squeeze(0).permute(
                1, 2, 0
            )  # C, H, W -> H, W, C
            output_frame = result_rgb_pt.cpu().numpy() / 255.0
            output_frames.append(torch.from_numpy(output_frame))

            # Process Alpha output
            result_alpha_pt = result_alpha_pt.squeeze(0).squeeze(
                0
            )  # 1, 1, H, W -> H, W
            output_mask = result_alpha_pt.cpu().numpy() / 255.0
            output_masks.append(torch.from_numpy(output_mask))

        output_frames_tensor = torch.stack(output_frames)
        output_masks_tensor = torch.stack(output_masks)

        return (output_frames_tensor, output_masks_tensor)


NODE_CLASS_MAPPINGS = {"ConvertToPixelArt": ConvertToPixelArt}

NODE_DISPLAY_NAME_MAPPINGS = {"ConvertToPixelArt": "Convert to Pixel Art"}
