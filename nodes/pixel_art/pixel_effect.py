import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelEffectModule(nn.Module):
    # Width of the smooth opacity transition on either side of
    # alpha_threshold -- see the note in forward() for why this replaced a
    # hard binary threshold.
    ALPHA_TRANSITION_SOFTNESS = 0.12

    # Minimum alpha-weighted support (in units of "fully-opaque source
    # pixels") the winning color family needs within a kernel window before
    # its own unmultiplied color is trusted -- see the note above
    # RELIABLE_COLOR_FLOOR's use in forward().
    RELIABLE_COLOR_FLOOR = 2.0

    # Minimum share of the block's TOTAL alpha coverage (all families
    # combined) the winning family needs to hold before it's trusted to
    # represent the whole block. dominant_vote_weight's saturation bias can
    # make a thin, uniformly-colored detail (e.g. a fold/seam line) win the
    # argmax against a much larger neutral area that's fragmented across
    # several tone-level bins -- no single one of those fragments looks
    # big, but their combined area is still the true majority.
    RELIABLE_AREA_SHARE_FLOOR = 0.35

    # Minimum TOTAL alpha coverage (all families combined, in units of
    # "fully-opaque source pixels") a block's own kernel window needs
    # before its local content is trusted at all -- independent of how
    # cleanly one family won that content. At a silhouette tip, a block's
    # kernel window can contain nothing but a sliver of dark outline stroke
    # and otherwise-transparent background: the outline family then wins
    # 100% of a real but tiny sample, clearing RELIABLE_AREA_SHARE_FLOOR
    # trivially even though there was never enough content nearby to draw
    # a real conclusion from. Below this floor, always defer to the wider
    # fallback search regardless of area_share.
    RELIABLE_TOTAL_COVERAGE_FLOOR = 4.0

    # How much wider than param_kernel_size the fallback color's own search
    # window is (see the note above its use in forward()).
    FALLBACK_KERNEL_MULTIPLIER = 2

    # saturation = chroma/cmax is scale-invariant, so it's numerically
    # meaningless for near-black colors: e.g. RGB (7,3,5), a typical dark
    # ink outline pixel, computes ~57% saturation -- HIGHER than a genuine
    # mid-tone lavender fill color's ~26% -- purely because both chroma and
    # cmax are tiny, not because the color is actually vivid. That lets
    # thin black outline strokes systematically outvote and out-bin larger
    # true-color fill areas. Below this cmax (as a 0-1 fraction), the raw
    # ratio is ramped toward 0 instead of trusted outright -- see
    # _damped_saturation().
    SATURATION_VALUE_FLOOR = 0.12

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

    def _damped_saturation(self, chroma, cmax, eps):
        """chroma/cmax, ramped toward 0 as cmax drops below
        SATURATION_VALUE_FLOOR -- see the constant's docstring."""
        raw = torch.where(cmax > eps, chroma / (cmax + eps), torch.zeros_like(chroma))
        ramp = (cmax / self.SATURATION_VALUE_FLOOR).clamp(0.0, 1.0)
        return raw * ramp

    def color_family_bin_idx(self, rgb, param_num_bins):
        """
        Build hue+tone bins:
        - chromatic pixels are split by hue family and luminance tone
        - near-neutral pixels are split by luminance tone only
        This keeps pixel-art shading/detail while still separating white/grey
        highlights from saturated hair colors.
        """
        r = rgb[0, 0] / 255.0
        g = rgb[0, 1] / 255.0
        b = rgb[0, 2] / 255.0

        eps = 1e-6
        cmax = torch.max(torch.stack([r, g, b], dim=0), dim=0).values
        cmin = torch.min(torch.stack([r, g, b], dim=0), dim=0).values
        chroma = cmax - cmin
        saturation = self._damped_saturation(chroma, cmax, eps)

        scale = int(np.ceil(np.sqrt(max(1, param_num_bins))))
        tone_levels = max(2, min(8, scale + 1))
        hue_families = max(3, min(6, scale))

        # Opponent-color angle for hue families.
        u = r - g
        v = (0.5 * (r + g)) - b
        hue_angle = torch.atan2(v, u + eps)  # [-pi, pi]
        hue_norm = (hue_angle + np.pi) / (2.0 * np.pi)
        hue_idx = (hue_norm * hue_families).long().clamp(0, hue_families - 1)

        # Tone is based on value/chroma peak to retain local shading.
        tone = cmax.clamp(0.0, 1.0)
        tone_idx = (tone * tone_levels).long().clamp(0, tone_levels - 1)

        chromatic_idx = hue_idx * tone_levels + tone_idx
        neutral_idx = (hue_families * tone_levels) + tone_idx

        sat_threshold = 0.10
        idx = torch.where(saturation >= sat_threshold, chromatic_idx, neutral_idx)
        num_bins = (hue_families + 1) * tone_levels
        neutral_start = hue_families * tone_levels
        return idx, num_bins, neutral_start

    def dominant_vote_weight(self, rgb, alpha_norm):
        """
        Build a per-pixel vote strength that favors saturated/chromatic colors
        over bright near-neutral highlights, without flattening detail.
        """
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

        eps = 1e-6
        cmax = torch.max(torch.cat([r, g, b], dim=1), dim=1, keepdim=True).values
        cmin = torch.min(torch.cat([r, g, b], dim=1), dim=1, keepdim=True).values
        chroma = cmax - cmin
        saturation = self._damped_saturation(chroma, cmax, eps)

        bright_neutral = (1.0 - saturation) * cmax
        vote_strength = (
            1.0
            + 0.75 * torch.sqrt(saturation.clamp(0.0, 1.0))
            + 0.35 * torch.sqrt(chroma.clamp(0.0, 1.0))
            - 0.30 * torch.pow(bright_neutral.clamp(0.0, 1.0), 1.20)
        )
        vote_strength = vote_strength.clamp(min=0.25, max=3.0)
        return alpha_norm * vote_strength

    def forward(
        self,
        rgb,
        alpha,
        param_num_bins,
        param_kernel_size,
        param_pixel_size,
        alpha_threshold=0.95,
        vote_state=None,
        prev_argmax=None,
        vote_beta=0.0,
        hysteresis_margin=0.0,
        vote_boost=None,
        prev_reliable=None,
        area_share_band=0.0,
    ):
        """
        Process RGB with alpha channel awareness.
        - RGB is padded with replicate (extends edge colors)
        - Alpha is padded with replicate (prevents edge darkening)
        - RGB is only output where alpha supports it

        Temporal stabilization (video): the block color is chosen by a
        winner-take-all argmax over color-family votes. On video, tiny
        frame-to-frame noise can flip which family wins a block even where
        the source is static, which reads as flickering blocks. Two
        mechanisms damp this, both operating on the *selection* rather than
        blurring the input:
        - vote_beta: EMA of the per-block vote totals across frames
          (vote_state carries the previous frame's smoothed votes).
        - hysteresis_margin: the previously winning family (prev_argmax)
          keeps a block unless a challenger's vote exceeds the incumbent's
          by this relative margin.
        A third mechanism damps a separate decision: whether the winning
        family's own color is trusted or a pooled-average fallback is used
        (see RELIABLE_AREA_SHARE_FLOOR below). Without damping this can
        flip every frame as a block's area_share drifts across the floor,
        alternating between two quite different colors even where nothing
        meaningfully changed -- area_share_band widens that floor into a
        band, Schmitt-trigger style, using the previous frame's reliable
        state (prev_reliable).
        Returns (result_rgb, result_alpha, vote_state, argmax, reliable);
        pass the last three back in for the next frame. For single images
        leave the defaults -- behavior is unchanged.
        """
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

        alpha_norm = alpha / 255.0

        # Build chroma-family bins and cast a weighted dominant-color vote.
        bin_idx, num_bins, neutral_start = self.color_family_bin_idx(
            rgb, param_num_bins
        )
        color_mask = self.create_mask_by_idx(bin_idx, max_z=num_bins)
        color_mask = torch.permute(color_mask, dims=[2, 0, 1]).unsqueeze(dim=0)

        alpha_weighted_mask = alpha_norm.repeat(1, num_bins, 1, 1) * color_mask
        vote_weight = self.dominant_vote_weight(rgb, alpha_norm)
        if vote_boost is not None:
            # Optional per-pixel vote multiplier (1, 1, H, W). Lets callers
            # amplify pixels that must not lose their block's winner-take-
            # all vote despite being an area minority -- e.g. thin, high-
            # contrast details like stems and outlines, which a pure area
            # vote erases.
            vote_weight = vote_weight * vote_boost
        vote_mask = vote_weight.repeat(1, num_bins, 1, 1) * color_mask

        # Weighted RGB accumulators per color family.
        r_weighted = r * alpha_weighted_mask
        g_weighted = g * alpha_weighted_mask
        b_weighted = b * alpha_weighted_mask

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
        alpha_weighted_mask_padded = F.pad(
            alpha_weighted_mask,
            (pad_size, pad_size, pad_size, pad_size),
            mode="replicate",
        )
        vote_mask_padded = F.pad(
            vote_mask, (pad_size, pad_size, pad_size, pad_size), mode="replicate"
        )

        # Pad alpha with replicate mode to prevent darkening at the edges
        alpha_norm_padded = F.pad(
            alpha_norm.repeat(1, num_bins, 1, 1),
            (pad_size, pad_size, pad_size, pad_size),
            mode="replicate",
        )

        kernel_conv = torch.ones(
            [num_bins, 1, param_kernel_size, param_kernel_size],
            device=rgb.device,
            dtype=rgb.dtype,
        )

        # Convolve all channels
        r_conv = F.conv2d(
            input=r_weighted_padded,
            weight=kernel_conv,
            padding=0,
            stride=param_pixel_size,
            groups=num_bins,
            bias=None,
        )[0, :, :, :]
        g_conv = F.conv2d(
            input=g_weighted_padded,
            weight=kernel_conv,
            padding=0,
            stride=param_pixel_size,
            groups=num_bins,
            bias=None,
        )[0, :, :, :]
        b_conv = F.conv2d(
            input=b_weighted_padded,
            weight=kernel_conv,
            padding=0,
            stride=param_pixel_size,
            groups=num_bins,
            bias=None,
        )[0, :, :, :]
        alpha_conv = F.conv2d(
            input=alpha_weighted_mask_padded,
            weight=kernel_conv,
            padding=0,
            stride=param_pixel_size,
            groups=num_bins,
            bias=None,
        )[0, :, :, :]
        vote_conv = F.conv2d(
            input=vote_mask_padded,
            weight=kernel_conv,
            padding=0,
            stride=param_pixel_size,
            groups=num_bins,
            bias=None,
        )[0, :, :, :]

        alpha_coverage_conv = F.conv2d(
            input=alpha_norm_padded,
            weight=kernel_conv,
            padding=0,
            stride=param_pixel_size,
            groups=num_bins,
            bias=None,
        )[0, :, :, :]

        # Pick the dominant color family by vote strength, with optional
        # temporal damping of the winner-take-all choice (see docstring).
        vote_used = vote_conv
        if (
            vote_state is not None
            and vote_beta > 0.0
            and vote_state.shape == vote_conv.shape
        ):
            vote_used = vote_beta * vote_state + (1.0 - vote_beta) * vote_conv

        _, alpha_argmax = torch.max(vote_used, dim=0)

        if (
            prev_argmax is not None
            and hysteresis_margin > 0.0
            and prev_argmax.shape == alpha_argmax.shape
        ):
            vote_hw = torch.permute(vote_used, dims=[1, 2, 0])
            challenger_vote = vote_used.max(dim=0).values
            incumbent_vote = self.select_by_idx(vote_hw, prev_argmax)
            # Keep the incumbent unless the challenger clearly beats it.
            # An incumbent with (near-)zero support has genuinely lost the
            # block (content moved away) and must not be kept.
            keep = (challenger_vote <= incumbent_vote * (1.0 + hysteresis_margin)) & (
                incumbent_vote > 1e-6
            )
            alpha_argmax = torch.where(keep, prev_argmax, alpha_argmax)
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

        # Unmultiply dominant-family RGB by alpha to get the final color.
        # This ratio is only meaningful when the winning family actually has
        # real support in this block; at a thin edge block where it barely
        # edged out the other families (e.g. one or two anti-aliased fringe
        # pixels), alpha_max is tiny and this collapses toward whatever that
        # handful of pixels happened to be -- often a dark outline/fringe
        # color, not the block's true content. That's normally hidden by
        # the correspondingly low result_alpha, but edge_style="hard" in
        # the caller can still promote such a block to fully opaque, making
        # the bad color a visible stray pixel. Below RELIABLE_COLOR_FLOOR,
        # fall back to the alpha-weighted average color across *all* bins
        # (i.e. ignoring the family split), which reflects the block's
        # actual dominant content instead of a barely-won sliver.
        r_final_selected = r_selected / (alpha_max + epsilon)
        g_final_selected = g_selected / (alpha_max + epsilon)
        b_final_selected = b_selected / (alpha_max + epsilon)

        # A block's own kernel window can be almost entirely transparent --
        # e.g. right at the tip of a silhouette curve, where only a sliver
        # of antialiased outline stroke falls inside it at all. Pooling
        # within that same starved window (as RELIABLE_COLOR_FLOOR/
        # RELIABLE_AREA_SHARE_FLOOR above do) still has nothing real to
        # average toward and just reproduces the same dark sliver. The
        # fallback color instead searches a wider neighborhood -- still
        # centered on this block, same stride/output grid, just a bigger
        # receptive field -- so it can draw on genuine nearby content (e.g.
        # the actual limb color a few pixels further in).
        fallback_kernel_size = param_kernel_size * self.FALLBACK_KERNEL_MULTIPLIER + 1
        fallback_pad_size = (fallback_kernel_size - 1) // 2
        fallback_kernel_conv = torch.ones(
            [1, 1, fallback_kernel_size, fallback_kernel_size],
            device=rgb.device,
            dtype=rgb.dtype,
        )

        def _wide_conv(channel):
            padded = F.pad(
                channel * alpha_norm,
                (fallback_pad_size, fallback_pad_size, fallback_pad_size, fallback_pad_size),
                mode="replicate",
            )
            return F.conv2d(
                padded, weight=fallback_kernel_conv, padding=0, stride=param_pixel_size
            )[0, 0, :, :]

        r_all = _wide_conv(r)
        g_all = _wide_conv(g)
        b_all = _wide_conv(b)
        alpha_coverage_wide = F.conv2d(
            F.pad(
                alpha_norm,
                (fallback_pad_size, fallback_pad_size, fallback_pad_size, fallback_pad_size),
                mode="replicate",
            ),
            weight=fallback_kernel_conv,
            padding=0,
            stride=param_pixel_size,
        )[0, 0, :, :]

        r_final_fallback = r_all / (alpha_coverage_wide + epsilon)
        g_final_fallback = g_all / (alpha_coverage_wide + epsilon)
        b_final_fallback = b_all / (alpha_coverage_wide + epsilon)

        area_share = alpha_max / (alpha_coverage + epsilon)
        if (
            prev_reliable is not None
            and area_share_band > 0.0
            and prev_reliable.shape == area_share.shape
        ):
            lo = self.RELIABLE_AREA_SHARE_FLOOR - area_share_band
            hi = self.RELIABLE_AREA_SHARE_FLOOR + area_share_band
            area_reliable = torch.where(prev_reliable, area_share >= lo, area_share >= hi)
        else:
            area_reliable = area_share >= self.RELIABLE_AREA_SHARE_FLOOR
        reliable = (
            (alpha_max >= self.RELIABLE_COLOR_FLOOR)
            & area_reliable
            & (alpha_coverage >= self.RELIABLE_TOTAL_COVERAGE_FLOOR)
        )
        r_final = torch.where(reliable, r_final_selected, r_final_fallback)
        g_final = torch.where(reliable, g_final_selected, g_final_fallback)
        b_final = torch.where(reliable, b_final_selected, b_final_fallback)

        # Build result RGB
        result_rgb = torch.stack([r_final, g_final, b_final], dim=-1)
        result_rgb = torch.permute(result_rgb, dims=[2, 0, 1]).unsqueeze(dim=0)
        result_rgb = F.interpolate(result_rgb, scale_factor=param_pixel_size)

        # Calculate alpha density
        kernel_area = param_kernel_size * param_kernel_size
        alpha_density = alpha_coverage / kernel_area

        # Bias alpha density smoothly around alpha_threshold instead of
        # hard-cutting it to a binary opaque/transparent mask. A pixel-art
        # block that's mostly-but-not-fully covered by source content
        # should come out partially transparent, reflecting how much of it
        # is actually covered -- forcing every block to be either fully
        # opaque or fully transparent throws that coverage information away
        # and gives harder, less faithful edges than the source alpha
        # actually has. alpha_threshold still acts as the center of the
        # transition (a block right at that coverage level lands at ~50%
        # alpha); ALPHA_TRANSITION_SOFTNESS controls how wide that graded
        # zone is on either side of it.
        if alpha_threshold > 0:
            lo = max(0.0, alpha_threshold - self.ALPHA_TRANSITION_SOFTNESS)
            hi = min(1.0, alpha_threshold + self.ALPHA_TRANSITION_SOFTNESS)
            t = ((alpha_density - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
            alpha_density = t * t * (3 - 2 * t)  # smoothstep

        result_alpha = alpha_density * 255.0
        result_alpha = result_alpha.unsqueeze(0).unsqueeze(0)
        result_alpha = F.interpolate(result_alpha, scale_factor=param_pixel_size)

        # RGB is returned straight (NOT premultiplied by alpha). Earlier
        # versions multiplied the graded alpha into RGB here, which baked
        # black into every partially-covered block -- downstream nodes and
        # encoders that composite using the alpha channel then darkened
        # those pixels a second time. Transparency lives solely in
        # result_alpha; result_rgb is the true block color everywhere the
        # dominant bin had any support.

        return result_rgb, result_alpha, vote_used, alpha_argmax, reliable
