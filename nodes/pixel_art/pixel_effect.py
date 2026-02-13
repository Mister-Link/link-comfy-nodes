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
        device = idx_z.device
        idx_x = torch.arange(h, device=device).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w, device=device).view([1, w]).repeat([h, 1])
        mask = torch.zeros([h, w, max_z], device=device, dtype=torch.float32)
        mask[idx_x, idx_y, idx_z] = 1
        return mask

    def select_by_idx(self, data, idx_z):
        h, w = idx_z.shape
        device = idx_z.device
        idx_x = torch.arange(h, device=device).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w, device=device).view([1, w]).repeat([h, 1])
        return data[idx_x, idx_y, idx_z]

    def rgb_to_bin_idx(self, rgb, param_num_bins, method="intensity"):
        """
        Compute a per-pixel bin index in [0, param_num_bins) used to find the
        dominant color cluster within each output block.

        method="intensity": bins by mean brightness (original behaviour, gives
            an oil-painting feel but confuses colors that share brightness).
        method="hue": bins by HSV hue for saturated pixels, falls back to
            binning by luminance for near-grey/white/black pixels.  This keeps
            orange and white in completely different bins regardless of how
            many bins are used.
        """
        # rgb shape: [1, C, H, W] with values in [0, 255]
        r = rgb[0, 0] / 255.0  # [H, W]
        g = rgb[0, 1] / 255.0
        b = rgb[0, 2] / 255.0

        if method == "intensity":
            val = (r + g + b) / 3.0  # mean brightness in [0, 1]
            idx = (val * param_num_bins).long().clamp(0, param_num_bins - 1)
            return idx  # [H, W]

        # --- hue method ---
        cmax = torch.max(torch.stack([r, g, b], dim=0), dim=0).values
        cmin = torch.min(torch.stack([r, g, b], dim=0), dim=0).values
        delta = cmax - cmin  # chroma

        # Hue in [0, 1)
        eps = 1e-6
        hue = torch.zeros_like(r)
        # Red is max
        mask_r = (cmax == r) & (delta > eps)
        hue[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + eps)) % 6.0
        # Green is max
        mask_g = (cmax == g) & (delta > eps)
        hue[mask_g] = (b[mask_g] - r[mask_g]) / (delta[mask_g] + eps) + 2.0
        # Blue is max
        mask_b = (cmax == b) & (delta > eps)
        hue[mask_b] = (r[mask_b] - g[mask_b]) / (delta[mask_b] + eps) + 4.0
        hue = (hue / 6.0).clamp(0.0, 1.0 - eps)  # normalise to [0, 1)

        saturation = torch.where(
            cmax > eps, delta / (cmax + eps), torch.zeros_like(delta)
        )

        # For low-saturation pixels (grey/white/black) hue is meaningless —
        # use the top quarter of bins for luminance-based discrimination.
        # Saturated pixels use the lower three quarters for hue bins.
        sat_threshold = 0.15
        num_hue_bins = max(1, int(param_num_bins * 0.75))
        num_lum_bins = param_num_bins - num_hue_bins  # at least 1 if num_bins >= 2

        hue_idx = (hue * num_hue_bins).long().clamp(0, num_hue_bins - 1)
        lum_idx = (cmax * num_lum_bins).long().clamp(
            0, max(0, num_lum_bins - 1)
        ) + num_hue_bins

        idx = torch.where(saturation >= sat_threshold, hue_idx, lum_idx)
        return idx  # [H, W]

    def forward(
        self,
        rgb,
        alpha,
        param_num_bins,
        param_kernel_size,
        param_pixel_size,
        allow_bleeding=True,
        alpha_threshold=0.95,
        dominance_threshold=0.72,
        outlier_filter=True,
        outlier_color_delta_threshold=72.0,
        bin_method="intensity",
    ):
        """
        Process RGB with alpha channel awareness.
        - RGB is padded with replicate (extends edge colors)
        - Alpha is padded with replicate (prevents edge darkening)
        - RGB is only output where alpha supports it
        """
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

        alpha_norm = alpha / 255.0

        # Compute per-pixel bin index using the chosen method
        bin_idx = self.rgb_to_bin_idx(rgb, param_num_bins, method=bin_method)
        intensity = self.create_mask_by_idx(bin_idx, max_z=param_num_bins)
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

        # Pad alpha with replicate mode to prevent darkening at the edges
        alpha_norm_padded = F.pad(
            alpha_norm.repeat(1, param_num_bins, 1, 1),
            (pad_size, pad_size, pad_size, pad_size),
            mode="replicate",
        )

        kernel_conv = torch.ones(
            [param_num_bins, 1, param_kernel_size, param_kernel_size],
            device=rgb.device,
            dtype=rgb.dtype,
        )
        kernel_conv_single = torch.ones(
            [1, 1, param_kernel_size, param_kernel_size],
            device=rgb.device,
            dtype=rgb.dtype,
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

        # Alpha-weighted color average per block across all intensities.
        r_total_padded = F.pad(
            r * alpha_norm, (pad_size, pad_size, pad_size, pad_size), mode="replicate"
        )
        g_total_padded = F.pad(
            g * alpha_norm, (pad_size, pad_size, pad_size, pad_size), mode="replicate"
        )
        b_total_padded = F.pad(
            b * alpha_norm, (pad_size, pad_size, pad_size, pad_size), mode="replicate"
        )
        alpha_norm_single_padded = F.pad(
            alpha_norm, (pad_size, pad_size, pad_size, pad_size), mode="replicate"
        )

        r_total_conv = F.conv2d(
            input=r_total_padded,
            weight=kernel_conv_single,
            padding=0,
            stride=param_pixel_size,
            groups=1,
            bias=None,
        )[0, 0, :, :]
        g_total_conv = F.conv2d(
            input=g_total_padded,
            weight=kernel_conv_single,
            padding=0,
            stride=param_pixel_size,
            groups=1,
            bias=None,
        )[0, 0, :, :]
        b_total_conv = F.conv2d(
            input=b_total_padded,
            weight=kernel_conv_single,
            padding=0,
            stride=param_pixel_size,
            groups=1,
            bias=None,
        )[0, 0, :, :]
        alpha_total_conv = F.conv2d(
            input=alpha_norm_single_padded,
            weight=kernel_conv_single,
            padding=0,
            stride=param_pixel_size,
            groups=1,
            bias=None,
        )[0, 0, :, :]

        # Select the dominant intensity bin.
        # Instead of raw argmax (which lets a small bright cluster "steal" the
        # result from a larger but spread-out region), normalise each bin's
        # accumulated alpha by the number of bins so that a minority of
        # high-intensity pixels cannot outweigh a majority that happens to be
        # distributed across several adjacent bins.
        #
        # We do this by computing a smoothed version of alpha_conv where each
        # bin's score is averaged with its immediate neighbors (triangular
        # window of width 3).  This merges adjacent orange-hair bins so their
        # combined weight beats the isolated white bin.
        alpha_conv_permuted_for_smooth = torch.permute(alpha_conv, dims=[1, 2, 0])
        # alpha_conv_permuted_for_smooth: [H, W, num_bins]
        # Pad the bin dimension with edge replication
        alpha_smooth = alpha_conv_permuted_for_smooth.clone()
        alpha_smooth[:, :, 1:-1] = (
            alpha_conv_permuted_for_smooth[:, :, 0:-2] * 0.25
            + alpha_conv_permuted_for_smooth[:, :, 1:-1] * 0.50
            + alpha_conv_permuted_for_smooth[:, :, 2:] * 0.25
        )
        alpha_smooth = torch.permute(alpha_smooth, dims=[2, 0, 1])
        _, alpha_argmax = torch.max(alpha_smooth, dim=0)
        # Use raw alpha_conv values (not smoothed) for the dominant bin's weight,
        # so that the dominance ratio stays comparable with alpha_total_conv.
        alpha_max = self.select_by_idx(
            torch.permute(alpha_conv, dims=[1, 2, 0]), alpha_argmax
        )
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

        # Unmultiply dominant-bin RGB by alpha to get dominant color.
        r_dominant = r_selected / (alpha_max + epsilon)
        g_dominant = g_selected / (alpha_max + epsilon)
        b_dominant = b_selected / (alpha_max + epsilon)

        # Also compute alpha-weighted block mean color across all intensities.
        r_mean = r_total_conv / (alpha_total_conv + epsilon)
        g_mean = g_total_conv / (alpha_total_conv + epsilon)
        b_mean = b_total_conv / (alpha_total_conv + epsilon)

        # If one bin is not strongly dominant, blend toward mean color to suppress
        # stray outlier colors (for example tiny leftover background patches).
        if dominance_threshold > 0:
            dominance_ratio = alpha_max / (alpha_total_conv + epsilon)
            fallback_mix = torch.clamp(
                (dominance_threshold - dominance_ratio)
                / (dominance_threshold + epsilon),
                0.0,
                1.0,
            )
        else:
            fallback_mix = torch.zeros_like(alpha_max)

        r_final = r_dominant * (1.0 - fallback_mix) + r_mean * fallback_mix
        g_final = g_dominant * (1.0 - fallback_mix) + g_mean * fallback_mix
        b_final = b_dominant * (1.0 - fallback_mix) + b_mean * fallback_mix

        if outlier_filter:
            # Suppress isolated color outlier blocks inside opaque, coherent regions.
            kernel_area = float(param_kernel_size * param_kernel_size)
            alpha_block = (
                (alpha_total_conv / kernel_area)
                .clamp(0.0, 1.0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            color_lowres = torch.stack([r_final, g_final, b_final], dim=0).unsqueeze(0)

            neighbor_kernel = torch.tensor(
                [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                device=rgb.device,
                dtype=rgb.dtype,
            ).view(1, 1, 3, 3)
            neighbor_kernel_rgb = neighbor_kernel.expand(3, 1, 3, 3)

            neighbor_alpha = F.conv2d(alpha_block, neighbor_kernel, padding=1)
            neighbor_color_sum = F.conv2d(
                color_lowres * alpha_block, neighbor_kernel_rgb, padding=1, groups=3
            )
            neighbor_mean = neighbor_color_sum / (neighbor_alpha + epsilon)

            color_delta = torch.sqrt(
                ((color_lowres - neighbor_mean) * (color_lowres - neighbor_mean)).sum(
                    dim=1, keepdim=True
                )
            )

            # Count similar opaque neighbors so only isolated outliers get replaced.
            # Also track the most popular neighbor color (the one with the most
            # look-alike neighbors), so we replace with an actual palette color
            # rather than a blended mean.
            _, _, h_low, w_low = color_lowres.shape
            color_padded = F.pad(color_lowres, (1, 1, 1, 1), mode="replicate")
            alpha_padded = F.pad(alpha_block, (1, 1, 1, 1), mode="replicate")
            similar_neighbor_count = torch.zeros_like(alpha_block)
            similarity_threshold = 42.0

            # best_replacement: the neighbor color whose own neighborhood is most
            # internally consistent (highest similar-neighbor count among neighbors).
            best_replacement = neighbor_mean.clone()
            best_replacement_score = torch.full_like(alpha_block, -1.0)

            for off_y in range(3):
                for off_x in range(3):
                    if off_x == 1 and off_y == 1:
                        continue
                    neighbor_color = color_padded[
                        :, :, off_y : off_y + h_low, off_x : off_x + w_low
                    ]
                    neighbor_alpha_local = alpha_padded[
                        :, :, off_y : off_y + h_low, off_x : off_x + w_low
                    ]
                    neighbor_color_delta = torch.sqrt(
                        (
                            (color_lowres - neighbor_color)
                            * (color_lowres - neighbor_color)
                        ).sum(dim=1, keepdim=True)
                    )
                    is_similar = (
                        (neighbor_color_delta < similarity_threshold)
                        & (neighbor_alpha_local > 0.90)
                    ).to(alpha_block.dtype)
                    similar_neighbor_count = similar_neighbor_count + is_similar

                    # How many of *this neighbor's* own neighbors look like it?
                    # Approximate: count how many other neighbors are close to this one.
                    neighbor_self_score = torch.zeros_like(alpha_block)
                    for oy2 in range(3):
                        for ox2 in range(3):
                            if ox2 == 1 and oy2 == 1:
                                continue
                            other = color_padded[
                                :, :, oy2 : oy2 + h_low, ox2 : ox2 + w_low
                            ]
                            d = torch.sqrt(
                                (
                                    (neighbor_color - other) * (neighbor_color - other)
                                ).sum(dim=1, keepdim=True)
                            )
                            neighbor_self_score = neighbor_self_score + (
                                d < similarity_threshold
                            ).to(alpha_block.dtype)

                    update = neighbor_self_score > best_replacement_score
                    best_replacement = torch.where(
                        update.expand_as(best_replacement),
                        neighbor_color,
                        best_replacement,
                    )
                    best_replacement_score = torch.where(
                        update, neighbor_self_score, best_replacement_score
                    )

            replace_isolated = (
                (alpha_block > 0.90)
                & (neighbor_alpha > 6.0)
                & (color_delta > outlier_color_delta_threshold)
                & (similar_neighbor_count < 1.5)
            )
            color_lowres = torch.where(replace_isolated, best_replacement, color_lowres)

            r_final = color_lowres[0, 0, :, :]
            g_final = color_lowres[0, 1, :, :]
            b_final = color_lowres[0, 2, :, :]

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
