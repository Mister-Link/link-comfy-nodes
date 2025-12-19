import numpy as np
import torch
from PIL import Image


def convert_image_to_tensor(img, preserve_alpha=False):
    """
    Convert PIL Image to torch tensor.

    Args:
        img: PIL Image
        preserve_alpha: If True, preserve alpha channel as 4th channel. If False, convert to RGB.

    Returns:
        torch tensor of shape (1, C, H, W) where C is 3 (RGB) or 4 (RGBA)
    """
    if preserve_alpha and img.mode == "RGBA":
        img_np = np.array(img).astype(np.float32)
        # RGBA image - keep all 4 channels
        img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]
    else:
        # For RGBA, extract just RGB channels (discard alpha during processing)
        # The alpha will be handled separately
        if img.mode == "RGBA":
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")

        img_np = np.array(img).astype(np.float32)
        img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]

    img_pt = torch.from_numpy(img_np)
    return img_pt


def convert_tensor_to_image(img_pt, has_alpha=False):
    """
    Convert torch tensor to PIL Image.

    Args:
        img_pt: torch tensor of shape (1, C, H, W) where C is 3 (RGB) or 4 (RGBA)
        has_alpha: If True, last channel is alpha and output will be RGBA

    Returns:
        PIL Image (RGB or RGBA depending on has_alpha)
    """
    img_pt = img_pt[0, ...].permute(1, 2, 0)
    result_np = img_pt.cpu().numpy().astype(np.uint8)

    if has_alpha:
        return Image.fromarray(result_np, "RGBA")
    else:
        return Image.fromarray(result_np, "RGB")


def extract_alpha_channel(img):
    """
    Extract alpha channel from RGBA image.

    Args:
        img: PIL Image in RGBA mode

    Returns:
        PIL Image (single channel, grayscale) containing alpha, or None if no alpha
    """
    if img.mode == "RGBA":
        return img.split()[3]
    return None


def restore_alpha_channel(rgb_img, alpha_channel):
    """
    Restore alpha channel to RGB image.

    Args:
        rgb_img: PIL Image in RGB mode
        alpha_channel: PIL Image (single channel) containing alpha values

    Returns:
        PIL Image in RGBA mode
    """
    if alpha_channel is None:
        return rgb_img

    # Ensure alpha channel matches RGB size
    if alpha_channel.size != rgb_img.size:
        alpha_channel = alpha_channel.resize(rgb_img.size, Image.NEAREST)

    rgba_img = Image.new("RGBA", rgb_img.size)
    rgba_img.paste(rgb_img, (0, 0))
    rgba_img.putalpha(alpha_channel)
    return rgba_img


def compute_valid_alpha_mask(
    alpha_channel, param_kernel_size, param_pixel_size, param_num_bins, model
):
    """
    Compute a validity mask that indicates which pixels in the output are sampled
    purely from valid (opaque) content, not from the transparent background.

    A pixel is considered "invalid" (should be transparent) if the kernel used to
    compute it extended beyond the original image bounds and sampled transparent areas.

    Args:
        alpha_channel: PIL Image (single channel grayscale) with original alpha
        param_kernel_size: Kernel size used for pixelation
        param_pixel_size: Pixel size used for pixelation
        param_num_bins: Number of bins used for pixelation
        model: PixelEffectModule used for processing

    Returns:
        numpy array of shape (H, W) with values 0-255 indicating validity
        (255 = valid/opaque, 0 = invalid/should be transparent)
    """
    # Convert alpha to numpy - threshold to binary opaque/transparent
    alpha_np = np.array(alpha_channel).astype(np.float32)
    h, w = alpha_np.shape

    # Create binary mask: opaque areas are 1.0, transparent are 0.0
    opaque_mask = (alpha_np > 127.5).astype(np.float32)

    # For each output pixel position, determine if the kernel window contains
    # only opaque pixels from the original image
    kernel_half = (param_kernel_size - 1) // 2
    validity_map = np.zeros((h, w), dtype=np.uint8)

    # Iterate through output pixel positions (considering stride)
    for out_y in range(0, h, param_pixel_size):
        for out_x in range(0, w, param_pixel_size):
            # Calculate kernel window bounds in input space
            min_y = out_y - kernel_half
            max_y = out_y + kernel_half + 1
            min_x = out_x - kernel_half
            max_x = out_x + kernel_half + 1

            # Check if kernel window is entirely within image bounds AND all opaque
            if min_y >= 0 and max_y <= h and min_x >= 0 and max_x <= w:
                # Kernel stays within bounds - check if all pixels are opaque
                kernel_window = opaque_mask[min_y:max_y, min_x:max_x]
                if np.all(kernel_window > 0.5):
                    # All pixels in kernel are opaque - mark as valid
                    out_y_end = min(out_y + param_pixel_size, h)
                    out_x_end = min(out_x + param_pixel_size, w)
                    validity_map[out_y:out_y_end, out_x:out_x_end] = 255
            # If kernel extends beyond bounds, leave as 0 (invalid/transparent)

    return validity_map


def apply_validity_mask(alpha_img, validity_mask_np):
    """
    Apply validity mask to alpha channel by setting alpha to 0 where mask is 0.

    Args:
        alpha_img: PIL Image (grayscale) with pixelated alpha values
        validity_mask_np: numpy array with validity mask (0-255)

    Returns:
        PIL Image (grayscale) with adjusted alpha values
    """
    alpha_np = np.array(alpha_img).astype(np.float32)

    # Make sure dimensions match, resize validity mask if needed
    if alpha_np.shape != validity_mask_np.shape:
        # Resize validity mask to match alpha
        validity_mask_pil = Image.fromarray(validity_mask_np)
        validity_mask_pil = validity_mask_pil.resize(
            (alpha_img.width, alpha_img.height), Image.NEAREST
        )
        validity_mask_np = np.array(validity_mask_pil)

    # Apply mask: where validity is 0, set alpha to 0 (fully transparent)
    alpha_np = alpha_np * (validity_mask_np.astype(np.float32) / 255.0)
    alpha_np = np.clip(alpha_np, 0, 255).astype(np.uint8)

    return Image.fromarray(alpha_np, mode="L")
