"""SEGS-based detailer for video with dimension fixing, 5D tensor handling, and optional VACE temporal coherence."""

from __future__ import annotations

from typing import Any

import impact.core as core
import numpy as np
import torch

import comfy.latent_formats
import comfy.model_management
import comfy.samplers
import comfy.utils
import node_helpers


class DetailerByMask:
    """
    Complete replacement for DetailerForEachPipeForAnimateDiff with full NAG compatibility.

    This node combines:
    - SEGS crop region fixing (ensures divisibility by 64)
    - NAG-compatible upscaling (maintains divisibility during upscale)
    - 5D tensor normalization (prevents dimension errors)
    - Custom paste logic that handles video frames
    - Optional VACE integration for temporal coherence

    Use this instead of the standard DetailerForEachPipeForAnimateDiff when working
    with NAG-patched models to avoid autocast and dimension errors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_frames": ("IMAGE",),
                "segs": ("SEGS",),
                "guide_size": (
                    "FLOAT",
                    {"default": 512, "min": 64, "max": 8192, "step": 8},
                ),
                "guide_size_for": (
                    "BOOLEAN",
                    {"default": True, "label_on": "bbox", "label_off": "crop_region"},
                ),
                "max_size": (
                    "FLOAT",
                    {"default": 1024, "min": 64, "max": 8192, "step": 8},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (core.get_schedulers(),),
                "denoise": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01},
                ),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "basic_pipe": ("BASIC_PIPE",),
                "temporal_mode": (
                    ["frame_by_frame", "batch_all_frames", "vace"],
                    {"default": "frame_by_frame"},
                ),
            },
            "optional": {
                "noise_mask_feather": (
                    "INT",
                    {"default": 20, "min": 0, "max": 100, "step": 1},
                ),
                "vace_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "link/Impact"
    DESCRIPTION = "SEGS-based detailer for video with NAG compatibility, automatic crop fixing, 5D tensor handling, and optional VACE temporal coherence"

    @staticmethod
    def _fix_crop_region(
        crop_region: tuple, img_width: int, img_height: int, divisor: int
    ) -> tuple:
        """Fix crop region to be divisible by divisor and within image bounds."""
        x1, y1, x2, y2 = crop_region

        # Clamp to image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))

        width = x2 - x1
        height = y2 - y1

        # Round UP to nearest multiple of divisor to avoid shrinking mask region
        width = ((width + divisor - 1) // divisor) * divisor
        height = ((height + divisor - 1) // divisor) * divisor

        # Recalculate x2, y2 from center to expand evenly
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        x1 = center_x - width // 2
        x2 = x1 + width
        y1 = center_y - height // 2
        y2 = y1 + height

        # Handle overflow by shifting
        if x2 > img_width:
            shift = x2 - img_width
            x1 -= shift
            x2 -= shift
        if x1 < 0:
            shift = -x1
            x1 += shift
            x2 += shift

        if y2 > img_height:
            shift = y2 - img_height
            y1 -= shift
            y2 -= shift
        if y1 < 0:
            shift = -y1
            y1 += shift
            y2 += shift

        # Final clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        return (x1, y1, x2, y2)

    def execute(
        self,
        image_frames,
        segs,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        feather,
        basic_pipe,
        temporal_mode="frame_by_frame",
        noise_mask_feather=0,
        vace_strength=1.0,
    ):
        """Process SEGS with dimension fixing, 5D tensor handling, and optional VACE temporal coherence."""
        try:
            from impact import utils
            from impact.animatediff_nodes import SEGSDetailerForAnimateDiff
            from impact.core import SEG
            from impact.utils import tensor_gaussian_blur_mask
        except ImportError:
            raise ImportError(
                "Could not import from Impact Pack. Make sure ComfyUI-Impact-Pack is installed."
            )

        # Route to VACE mode if selected
        if temporal_mode == "vace":
            return self._execute_vace_mode(
                image_frames,
                segs,
                guide_size,
                guide_size_for,
                max_size,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                denoise,
                feather,
                basic_pipe,
                noise_mask_feather,
                vace_strength,
            )

        # Hardcoded NAG divisor
        nag_divisor = 64

        # Get image dimensions
        img_height = image_frames.shape[1]
        img_width = image_frames.shape[2]

        # Process SEGS directly
        header, seg_list = segs
        fixed_seg_list = []

        for i, seg in enumerate(seg_list):
            crop_region = getattr(seg, "crop_region", None)

            if crop_region is not None:
                fixed_crop = self._fix_crop_region(
                    crop_region, img_width, img_height, nag_divisor
                )

                if fixed_crop != crop_region:
                    print(
                        f"[Detailer by Mask] Seg {i}: Fixed crop_region {crop_region} → {fixed_crop} (divisor={nag_divisor})"
                    )

                # Replace seg with fixed crop
                if hasattr(seg, "_replace"):
                    fixed_seg = seg._replace(crop_region=fixed_crop)
                else:
                    # Fallback for non-namedtuple SEG
                    from impact.core import SEG

                    fixed_seg = SEG(
                        seg.cropped_image,
                        seg.cropped_mask,
                        seg.confidence,
                        fixed_crop,  # crop_region
                        seg.bbox,
                        seg.label,
                        seg.control_net_wrapper
                        if hasattr(seg, "control_net_wrapper")
                        else None,
                    )

                fixed_seg_list.append(fixed_seg)
            else:
                fixed_seg_list.append(seg)

        fixed_segs = (header, fixed_seg_list)

        # Create hook to ensure dimensions are divisible by 64
        dimension_hook = DivisibleDimensionHook(nag_divisor)

        # Process each segment using Impact Pack's approach
        from impact.segs_nodes import SEGSPaste

        enhanced_segs = []
        cnet_image_list = []

        model, clip, vae, positive, negative = basic_pipe

        total_segments = len(fixed_seg_list)
        for seg_idx, sub_seg in enumerate(fixed_seg_list):
            print(
                f"[Detailer by Mask] Processing segment {seg_idx + 1}/{total_segments}"
            )
            seg = sub_seg

            # Use the segment's pre-cropped image if available, otherwise crop from source
            if seg.cropped_image is not None:
                # Segment already has cropped image (e.g., from Mask to SEGS)
                cropped_image_frames = utils.to_tensor(seg.cropped_image)
                if isinstance(cropped_image_frames, torch.Tensor):
                    cropped_image_frames = cropped_image_frames.cpu().numpy()
                print(
                    f"[Detailer by Mask] Using pre-cropped image, shape: {cropped_image_frames.shape}"
                )
            else:
                # Need to crop from the full image_frames
                # Since we have 17 separate segments (one per frame), only process the corresponding frame
                # Use segment index to determine which frame to process
                if seg_idx < len(image_frames):
                    # Crop only the frame that corresponds to this segment
                    image = image_frames[seg_idx].unsqueeze(0)
                    cropped_image = utils.crop_tensor4(image, seg.crop_region)
                    cropped_image_frames = utils.to_tensor(cropped_image).cpu().numpy()
                    print(
                        f"[Detailer by Mask] Cropped frame {seg_idx} from full video, shape: {cropped_image_frames.shape}"
                    )
                else:
                    print(
                        f"[Detailer by Mask] ERROR: Segment index {seg_idx} out of range for {len(image_frames)} frames"
                    )
                    continue

            # Crop conditioning
            from impact import core
            from impact.core import SEG

            cropped_positive = [
                [
                    condition,
                    {
                        k: core.crop_condition_mask(
                            v, cropped_image_frames, seg.crop_region
                        )
                        if k == "mask"
                        else v
                        for k, v in details.items()
                    },
                ]
                for condition, details in positive
            ]

            cropped_negative = [
                [
                    condition,
                    {
                        k: core.crop_condition_mask(
                            v, cropped_image_frames, seg.crop_region
                        )
                        if k == "mask"
                        else v
                        for k, v in details.items()
                    },
                ]
                for condition, details in negative
            ]

            # Enhance detail with NAG hook
            # Detect if this is a single-frame segment or multi-frame batch
            is_single_frame = (
                isinstance(cropped_image_frames, np.ndarray)
                and cropped_image_frames.ndim == 4
                and cropped_image_frames.shape[0] == 1
            )
            print(
                f"[Detailer by Mask] cropped_image_frames shape: {cropped_image_frames.shape if isinstance(cropped_image_frames, np.ndarray) else 'not numpy'}, is_single_frame: {is_single_frame}"
            )

            if not (isinstance(model, str) and model == "DUMMY"):
                if is_single_frame:
                    # Single frame - use regular enhance_detail for better denoise behavior
                    cropped_image_tensor = torch.from_numpy(cropped_image_frames)
                    enhanced_image_tensor, cnet_images = core.enhance_detail(
                        cropped_image_tensor,
                        model,
                        clip,
                        vae,
                        guide_size,
                        guide_size_for,
                        max_size,
                        seg.bbox,
                        seed,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        cropped_positive,
                        cropped_negative,
                        denoise,
                        seg.cropped_mask,
                        force_inpaint=False,
                        refiner_ratio=None,
                        refiner_model=None,
                        refiner_clip=None,
                        refiner_positive=None,
                        refiner_negative=None,
                        control_net_wrapper=seg.control_net_wrapper
                        if hasattr(seg, "control_net_wrapper")
                        else None,
                        noise_mask_feather=noise_mask_feather,
                        scheduler_func=None,
                        detailer_hook=dimension_hook,
                    )
                else:
                    # Multi-frame batch - use animatediff version
                    enhanced_image_tensor, cnet_images = (
                        core.enhance_detail_for_animatediff(
                            cropped_image_frames,
                            model,
                            clip,
                            vae,
                            guide_size,
                            guide_size_for,
                            max_size,
                            seg.bbox,
                            seed,
                            steps,
                            cfg,
                            sampler_name,
                            scheduler,
                            cropped_positive,
                            cropped_negative,
                            denoise,
                            seg.cropped_mask,
                            refiner_ratio=None,
                            refiner_model=None,
                            refiner_clip=None,
                            refiner_positive=None,
                            refiner_negative=None,
                            control_net_wrapper=seg.control_net_wrapper
                            if hasattr(seg, "control_net_wrapper")
                            else None,
                            noise_mask_feather=noise_mask_feather,
                            scheduler_func=None,
                            detailer_hook=dimension_hook,
                        )
                    )
            else:
                enhanced_image_tensor = cropped_image_frames
                cnet_images = None

            if cnet_images is not None:
                cnet_image_list.extend(cnet_images)

            if enhanced_image_tensor is None:
                new_cropped_image = cropped_image_frames
            else:
                # Normalize to 4D if needed BEFORE converting to numpy
                if enhanced_image_tensor.ndim == 5:
                    # Flatten batch and frames dimensions: [B, F, H, W, C] -> [B*F, H, W, C]
                    b, f, h, w, c = enhanced_image_tensor.shape
                    enhanced_image_tensor = enhanced_image_tensor.reshape(
                        b * f, h, w, c
                    )
                    print(
                        f"[NAG Detailer] Normalized enhanced tensor from 5D {(b, f, h, w, c)} to 4D {enhanced_image_tensor.shape}"
                    )

                new_cropped_image = enhanced_image_tensor.cpu().numpy()

            new_seg = SEG(
                new_cropped_image,
                seg.cropped_mask,
                seg.confidence,
                seg.crop_region,
                seg.bbox,
                seg.label,
                None,
            )

            # Paste the enhanced segment back onto the SINGLE corresponding frame
            # We can't use SEGSPaste because it tries to paste to all frames
            if is_single_frame:
                # Manual single-frame paste
                from impact.utils import tensor_gaussian_blur_mask

                # Get the enhanced image as tensor
                if isinstance(new_cropped_image, np.ndarray):
                    ref_image = torch.from_numpy(new_cropped_image)
                else:
                    ref_image = new_cropped_image

                # Handle mask
                mask = seg.cropped_mask
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                if mask.ndim == 3:
                    mask = mask[0]  # Take first frame's mask

                mask = tensor_gaussian_blur_mask(mask, feather)

                # Get the single frame to modify
                frame_to_modify = image_frames[seg_idx].unsqueeze(0).clone()

                # Paste
                x, y, *_ = seg.crop_region
                ref_image = ref_image.to(frame_to_modify.device)
                mask = mask.to(frame_to_modify.device)
                utils.tensor_paste(frame_to_modify, ref_image, (x, y), mask)

                # Put the modified frame back
                image_frames[seg_idx] = frame_to_modify[0]
            else:
                # Multi-frame - use SEGSPaste
                enhanced_seg = (header, [new_seg])
                image_frames = SEGSPaste.doit(
                    image_frames, enhanced_seg, feather, alpha=255
                )[0]

            # Call hook's post_paste
            image_frames = dimension_hook.post_paste(image_frames)

            enhanced_segs.append(new_seg)

        # Collect all masks from enhanced segments
        output_mask = None
        for seg in enhanced_segs:
            if seg.cropped_mask is not None:
                if output_mask is None:
                    output_mask = seg.cropped_mask
                else:
                    # Combine masks if multiple segments
                    if isinstance(output_mask, torch.Tensor) and isinstance(
                        seg.cropped_mask, torch.Tensor
                    ):
                        output_mask = torch.max(output_mask, seg.cropped_mask)

        # If no mask found, create empty mask
        if output_mask is None:
            output_mask = torch.zeros((img_height, img_width), dtype=torch.float32)

        return (image_frames, output_mask)

    def _execute_vace_mode(
        self,
        image_frames,
        segs,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        feather,
        basic_pipe,
        noise_mask_feather,
        vace_strength,
    ):
        """
        VACE mode: Process all frames together using VACE for temporal coherence.

        Instead of processing frame-by-frame, this mode:
        1. Crops the face region from all frames
        2. Creates a combined mask from all SEGS
        3. Uses VACE conditioning to process all frames together
        4. Pastes the enhanced regions back

        This provides temporal coherence because VACE processes all frames together.
        """
        try:
            from impact import utils
            from impact.core import SEG
            from impact.utils import tensor_gaussian_blur_mask
        except ImportError:
            raise ImportError(
                "Could not import from Impact Pack. Make sure ComfyUI-Impact-Pack is installed."
            )

        nag_divisor = 64
        img_height = image_frames.shape[1]
        img_width = image_frames.shape[2]
        num_frames = image_frames.shape[0]

        print(
            f"[Detailer by Mask VACE] Processing {num_frames} frames with VACE temporal coherence"
        )

        model, clip, vae, positive, negative = basic_pipe

        # Process SEGS to fix crop regions
        header, seg_list = segs

        if len(seg_list) == 0:
            print("[Detailer by Mask VACE] No segments to process")
            output_mask = torch.zeros((img_height, img_width), dtype=torch.float32)
            return (image_frames, output_mask)

        # Find the unified bounding box across all segments
        # This will be used to crop all frames to the same region
        min_x1, min_y1 = float("inf"), float("inf")
        max_x2, max_y2 = 0, 0

        for seg in seg_list:
            if seg.crop_region is not None:
                x1, y1, x2, y2 = seg.crop_region
                min_x1 = min(min_x1, x1)
                min_y1 = min(min_y1, y1)
                max_x2 = max(max_x2, x2)
                max_y2 = max(max_y2, y2)

        # Fix the unified crop region to be divisible by nag_divisor
        unified_crop = self._fix_crop_region(
            (int(min_x1), int(min_y1), int(max_x2), int(max_y2)),
            img_width,
            img_height,
            nag_divisor,
        )
        x1, y1, x2, y2 = unified_crop
        crop_width = x2 - x1
        crop_height = y2 - y1

        print(
            f"[Detailer by Mask VACE] Unified crop region: {unified_crop} ({crop_width}x{crop_height})"
        )

        # Crop all frames to the unified region
        cropped_frames = image_frames[:, y1:y2, x1:x2, :]
        print(f"[Detailer by Mask VACE] Cropped frames shape: {cropped_frames.shape}")

        # Build the combined mask for all frames
        # Each segment corresponds to one frame
        combined_mask = torch.zeros(
            (num_frames, crop_height, crop_width), dtype=torch.float32
        )

        for seg_idx, seg in enumerate(seg_list):
            if seg_idx >= num_frames:
                break
            if seg.cropped_mask is not None:
                mask = seg.cropped_mask
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                if mask.ndim == 3:
                    mask = mask[0]  # Take first slice if 3D

                # The mask is for the segment's crop region, need to place it in unified crop
                seg_x1, seg_y1, seg_x2, seg_y2 = seg.crop_region
                # Offset within the unified crop
                offset_x = seg_x1 - x1
                offset_y = seg_y1 - y1
                mask_h, mask_w = mask.shape

                # Place mask in the combined mask at the right position
                end_y = min(offset_y + mask_h, crop_height)
                end_x = min(offset_x + mask_w, crop_width)
                mask_end_y = end_y - offset_y
                mask_end_x = end_x - offset_x

                if offset_y >= 0 and offset_x >= 0:
                    combined_mask[seg_idx, offset_y:end_y, offset_x:end_x] = mask[
                        :mask_end_y, :mask_end_x
                    ]
            else:
                # No mask for this segment - use full region
                combined_mask[seg_idx] = 1.0

        print(f"[Detailer by Mask VACE] Combined mask shape: {combined_mask.shape}")

        # Apply VACE conditioning to process all frames together
        # VACE encodes the control video and mask into the conditioning

        # Get latent dimensions
        latent_height = crop_height // 8
        latent_width = crop_width // 8
        latent_length = ((num_frames - 1) // 4) + 1

        # Encode the cropped frames
        control_video = cropped_frames.clone()

        # Prepare control video for VACE (shift to 0.5-centered for inactive regions)
        control_video_centered = control_video - 0.5

        # Expand mask for broadcasting: [F, H, W] -> [F, H, W, 1]
        mask_expanded = combined_mask.unsqueeze(-1)

        # inactive = regions we want to preserve (mask = 0)
        # reactive = regions we want to regenerate (mask = 1)
        inactive = (control_video_centered * (1 - mask_expanded)) + 0.5
        reactive = (control_video_centered * mask_expanded) + 0.5

        # Encode both through VAE
        inactive_latent = vae.encode(inactive[:, :, :, :3])
        reactive_latent = vae.encode(reactive[:, :, :, :3])

        # Concatenate for VACE format
        control_video_latent = torch.cat((inactive_latent, reactive_latent), dim=1)
        print(
            f"[Detailer by Mask VACE] Control video latent shape: {control_video_latent.shape}"
        )

        # Process mask for latent space
        vae_stride = 8
        mask_for_latent = combined_mask.unsqueeze(-1)  # [F, H, W, 1]
        mask_for_latent = mask_for_latent.view(
            num_frames, latent_height, vae_stride, latent_width, vae_stride
        )
        mask_for_latent = mask_for_latent.permute(2, 4, 0, 1, 3)
        mask_for_latent = mask_for_latent.reshape(
            vae_stride * vae_stride, num_frames, latent_height, latent_width
        )
        mask_for_latent = (
            torch.nn.functional.interpolate(
                mask_for_latent.unsqueeze(0),
                size=(latent_length, latent_height, latent_width),
                mode="nearest-exact",
            )
            .squeeze(0)
            .unsqueeze(0)
        )

        # Apply VACE conditioning
        vace_positive = node_helpers.conditioning_set_values(
            positive,
            {
                "vace_frames": [control_video_latent],
                "vace_mask": [mask_for_latent],
                "vace_strength": [vace_strength],
            },
            append=True,
        )
        vace_negative = node_helpers.conditioning_set_values(
            negative,
            {
                "vace_frames": [control_video_latent],
                "vace_mask": [mask_for_latent],
                "vace_strength": [vace_strength],
            },
            append=True,
        )

        # Create starting latent
        latent = torch.zeros(
            [1, 16, latent_length, latent_height, latent_width],
            device=comfy.model_management.intermediate_device(),
        )

        # Sample with VACE conditioning
        print(f"[Detailer by Mask VACE] Sampling with VACE conditioning...")

        from nodes import common_ksampler

        # Prepare latent dict
        latent_dict = {"samples": latent}

        # Use denoise to blend original with generated
        samples = common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            vace_positive,
            vace_negative,
            latent_dict,
            denoise=denoise,
        )[0]

        # Decode the result
        decoded = vae.decode(samples["samples"])
        print(f"[Detailer by Mask VACE] Decoded shape: {decoded.shape}")

        # The decoded tensor may have different frame count due to latent compression
        # Interpolate back to original frame count if needed
        if decoded.shape[0] != num_frames:
            # Reshape for interpolation: [F, H, W, C] -> [1, C, F, H, W]
            decoded_5d = decoded.permute(3, 0, 1, 2).unsqueeze(0)
            decoded_5d = torch.nn.functional.interpolate(
                decoded_5d,
                size=(num_frames, crop_height, crop_width),
                mode="trilinear",
                align_corners=False,
            )
            decoded = decoded_5d.squeeze(0).permute(1, 2, 3, 0)
            print(f"[Detailer by Mask VACE] Interpolated to {decoded.shape}")

        # Paste the enhanced regions back
        output_frames = image_frames.clone()

        # Apply feathered mask for smooth blending
        for frame_idx in range(num_frames):
            frame_mask = combined_mask[frame_idx]
            frame_mask = tensor_gaussian_blur_mask(frame_mask, feather)
            frame_mask = frame_mask.unsqueeze(-1)  # [H, W, 1]

            # Get the enhanced crop for this frame
            enhanced_crop = decoded[frame_idx]
            original_crop = output_frames[frame_idx, y1:y2, x1:x2, :]

            # Blend based on mask
            blended = original_crop * (1 - frame_mask) + enhanced_crop * frame_mask
            output_frames[frame_idx, y1:y2, x1:x2, :] = blended

        print(
            f"[Detailer by Mask VACE] Done - processed {num_frames} frames with temporal coherence"
        )

        # Return combined mask
        output_mask = combined_mask.max(dim=0)[0]  # Union of all frame masks

        return (output_frames, output_mask)


class DivisibleDimensionHook:
    """Ensures dimensions are divisible by a given value (e.g., 64 for NAG models)."""

    def __init__(self, divisor: int = 64):
        self.divisor = divisor

    def touch_scaled_size(self, width: int, height: int) -> tuple[int, int]:
        """Round dimensions to nearest multiple of divisor."""
        adjusted_width = (width // self.divisor) * self.divisor
        adjusted_height = (height // self.divisor) * self.divisor

        if adjusted_width < self.divisor:
            adjusted_width = self.divisor
        if adjusted_height < self.divisor:
            adjusted_height = self.divisor

        if adjusted_width != width or adjusted_height != height:
            print(
                f"[NAG Hook] Adjusted upscale from ({width}, {height}) to ({adjusted_width}, {adjusted_height})"
            )

        return adjusted_width, adjusted_height

    def get_custom_sampler(self):
        """Return custom sampler if needed - None for default."""
        return None

    def post_encode(self, latent: dict) -> dict:
        """Called after VAE encoding - passthrough."""
        return latent

    def pre_decode(self, latent: dict) -> dict:
        """Called before VAE decoding - passthrough."""
        return latent

    def post_decode(self, image):
        """Called after VAE decoding - passthrough."""
        return image

    def post_paste(self, image):
        """Called after pasting enhanced segment - passthrough."""
        return image

    def post_upscale(self, image, mask):
        """Called after upscaling - passthrough."""
        return image

    def get_skip_sampling(self):
        """Return whether to skip sampling - False means do sampling."""
        return False

    def set_steps(self, info):
        """Called to set step info - no-op."""
        pass

    def cycle_latent(self, latent):
        """Called each cycle - passthrough."""
        return latent

    def pre_ksample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise,
    ):
        """Called before ksampler - return unchanged values."""
        return (
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise,
        )

    def get_custom_noise(self, seed, noise, is_touched=False):
        """Return custom noise - None means use default."""
        return None, False

    def post_detection(self, segs):
        """Called after detection - passthrough."""
        return segs

    def post_crop_region(self, w, h, bbox, crop_region):
        """Called after crop region calculation - passthrough."""
        return crop_region


__all__ = ["DetailerByMask"]
