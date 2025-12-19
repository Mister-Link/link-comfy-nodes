import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class PixelEffectModule(nn.Module):
    def __init__(self):
        super(PixelEffectModule, self).__init__()

    def create_mask_by_idx(self, idx_z, max_z):
        h, w = idx_z.shape
        idx_x = torch.arange(h).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w).view([1, w]).repeat([h, 1])
        mask = torch.zeros([h, w, max_z])
        mask[idx_x, idx_y, idx_z] = 1
        return mask

    def select_by_idx(self, data, idx_z):
        h, w = idx_z.shape
        idx_x = torch.arange(h).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w).view([1, w]).repeat([h, 1])
        return data[idx_x, idx_y, idx_z]

    def forward(
        self,
        rgb,
        alpha,
        param_num_bins,
        param_kernel_size,
        param_pixel_size,
        allow_bleeding=True,
        alpha_threshold=0.95,
    ):
        """
        Process RGB with alpha channel awareness.
        - RGB is padded with replicate (extends edge colors)
        - Alpha is padded with zeros (true transparency at edges)
        - RGB is only output where alpha supports it
        """
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

        alpha_norm = alpha / 255.0

        # Calculate intensity from RGB
        intensity_idx = torch.mean(rgb, dim=[0, 1]) / 256.0 * param_num_bins
        intensity_idx = intensity_idx.long()
        intensity = self.create_mask_by_idx(intensity_idx, max_z=param_num_bins)
        intensity = torch.permute(intensity, dims=[2, 0, 1]).unsqueeze(dim=0)

        # Weight intensity by alpha for proper blending
        alpha_intensity = alpha_norm.repeat(1, param_num_bins, 1, 1) * intensity

        # Weight RGB by alpha
        r_weighted = r * alpha_intensity
        g_weighted = g * alpha_intensity
        b_weighted = b * alpha_intensity

        pad_size = (param_kernel_size - 1) // 2

        # Pad RGB with replicate mode (extends edge colors naturally)
        r_weighted_padded = F.pad(
            r_weighted, (pad_size, pad_size, pad_size, pad_size), mode="replicate"
        )
        g_weighted_padded = F.pad(
            g_weighted, (pad_size, pad_size, pad_size, pad_size), mode="replicate"
        )
        b_weighted_padded = F.pad(
            b_weighted, (pad_size, pad_size, pad_size, pad_size), mode="replicate"
        )
        alpha_intensity_padded = F.pad(
            alpha_intensity, (pad_size, pad_size, pad_size, pad_size), mode="replicate"
        )

        # Pad alpha itself with zeros (true transparency at edges)
        alpha_norm_padded = F.pad(
            alpha_norm.repeat(1, param_num_bins, 1, 1),
            (pad_size, pad_size, pad_size, pad_size),
            mode="constant",
            value=0.0,
        )

        kernel_conv = torch.ones(
            [param_num_bins, 1, param_kernel_size, param_kernel_size]
        )

        # Convolve all channels
        r_conv = F.conv2d(
            input=r_weighted_padded,
            weight=kernel_conv,
            padding=0,
            stride=param_pixel_size,
            groups=param_num_bins,
            bias=None,
        )[0, :, :, :]
        g_conv = F.conv2d(
            input=g_weighted_padded,
            weight=kernel_conv,
            padding=0,
            stride=param_pixel_size,
            groups=param_num_bins,
            bias=None,
        )[0, :, :, :]
        b_conv = F.conv2d(
            input=b_weighted_padded,
            weight=kernel_conv,
            padding=0,
            stride=param_pixel_size,
            groups=param_num_bins,
            bias=None,
        )[0, :, :, :]
        alpha_conv = F.conv2d(
            input=alpha_intensity_padded,
            weight=kernel_conv,
            padding=0,
            stride=param_pixel_size,
            groups=param_num_bins,
            bias=None,
        )[0, :, :, :]

        alpha_coverage_conv = F.conv2d(
            input=alpha_norm_padded,
            weight=kernel_conv,
            padding=0,
            stride=param_pixel_size,
            groups=param_num_bins,
            bias=None,
        )[0, :, :, :]

        # Select the dominant intensity bin
        alpha_max, alpha_argmax = torch.max(alpha_conv, dim=0)
        alpha_coverage_conv_permuted = torch.permute(
            alpha_coverage_conv, dims=[1, 2, 0]
        )
        alpha_coverage = self.select_by_idx(alpha_coverage_conv_permuted, alpha_argmax)

        # Select RGB from the dominant bin
        r_conv_permuted = torch.permute(r_conv, dims=[1, 2, 0])
        g_conv_permuted = torch.permute(g_conv, dims=[1, 2, 0])
        b_conv_permuted = torch.permute(b_conv, dims=[1, 2, 0])

        r_selected = self.select_by_idx(r_conv_permuted, alpha_argmax)
        g_selected = self.select_by_idx(g_conv_permuted, alpha_argmax)
        b_selected = self.select_by_idx(b_conv_permuted, alpha_argmax)

        epsilon = 1e-8

        # Unmultiply RGB by alpha (divide by alpha_max to get true colors)
        r_final = r_selected / (alpha_max + epsilon)
        g_final = g_selected / (alpha_max + epsilon)
        b_final = b_selected / (alpha_max + epsilon)

        # Build result RGB
        result_rgb = torch.stack([r_final, g_final, b_final], dim=-1)
        result_rgb = torch.permute(result_rgb, dims=[2, 0, 1]).unsqueeze(dim=0)
        result_rgb = F.interpolate(result_rgb, scale_factor=param_pixel_size)

        # Calculate alpha density
        kernel_area = param_kernel_size * param_kernel_size
        alpha_density = alpha_coverage / kernel_area

        # Threshold alpha for harsh pixel art edges
        if alpha_threshold > 0:
            alpha_density = (alpha_density > alpha_threshold).float()

        result_alpha = alpha_density * 255.0
        result_alpha = result_alpha.unsqueeze(0).unsqueeze(0)
        result_alpha = F.interpolate(result_alpha, scale_factor=param_pixel_size)

        # Discard nearly-transparent pixels entirely - they add noise without value
        # If a pixel is mostly transparent, it shouldn't be output at all
        alpha_mask = result_alpha / 255.0
        min_alpha_for_output = 0.1  # Discard pixels with <10% opacity

        # Where alpha is too low, force it to zero (completely transparent)
        result_alpha = torch.where(
            alpha_mask > min_alpha_for_output,
            result_alpha,
            torch.zeros_like(result_alpha),
        )

        # Mask RGB by the (possibly zeroed) alpha
        alpha_mask = result_alpha / 255.0
        result_rgb = result_rgb * alpha_mask

        return result_rgb, result_alpha


def test1():
    img = Image.open("../images/example_input_mountain.jpg").convert("RGB")
    img_np = np.array(img).astype(np.float32)
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]
    img_pt = torch.from_numpy(img_np)

    model = PixelEffectModule()
    model.eval()

    with torch.no_grad():
        result_rgb_pt = model(
            img_pt, param_num_bins=4, param_kernel_size=11, param_pixel_size=16
        )
        result_rgb_pt = result_rgb_pt[0, ...].permute(1, 2, 0)

    print("img_pt", img_pt.shape)
    print("result_rgb_pt", result_rgb_pt.shape)

    result_rgb_np = result_rgb_pt.cpu().numpy().astype(np.uint8)
    Image.fromarray(result_rgb_np).save("./test_result_pixel_effect.png")


if __name__ == "__main__":
    test1()
