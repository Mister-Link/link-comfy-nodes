"""
WanAnimateToVideo with per-frame mask control.

This node wraps WanAnimateToVideo and adds the ability to feed in masks for each frame,
so that latents for frames with black masks are not changed, while white ones are affected.
"""

import torch

import comfy.model_management
import comfy.utils
import nodes
from node_helpers import conditioning_set_values


class WanAnimateToVideoMasked:
    """
    WanAnimateToVideo with per-frame mask control.

    Allows you to provide masks that control which frames are affected by the animation.
    - Black mask (0.0) = frame latents unchanged (original content preserved)
    - White mask (1.0) = frame latents fully affected by animation
    - Gray values = blend between original and animated
    """

    CATEGORY = "conditioning/video_models"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "positive",
        "negative",
        "latent",
        "trim_latent",
        "trim_image",
        "video_frame_offset",
    )
    FUNCTION = "execute"
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": (
                    "INT",
                    {
                        "default": 832,
                        "min": 16,
                        "max": nodes.MAX_RESOLUTION,
                        "step": 16,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 480,
                        "min": 16,
                        "max": nodes.MAX_RESOLUTION,
                        "step": 16,
                    },
                ),
                "length": (
                    "INT",
                    {"default": 77, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "continue_motion_max_frames": (
                    "INT",
                    {"default": 5, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4},
                ),
                "video_frame_offset": (
                    "INT",
                    {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1},
                ),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "reference_image": ("IMAGE",),
                "face_video": ("IMAGE",),
                "pose_video": ("IMAGE",),
                "background_video": ("IMAGE",),
                "character_mask": ("MASK",),
                "continue_motion": ("IMAGE",),
                "frame_masks": ("MASK",),  # New input: per-frame animation masks
            },
        }

    def execute(
        self,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        continue_motion_max_frames,
        video_frame_offset,
        reference_image=None,
        clip_vision_output=None,
        face_video=None,
        pose_video=None,
        continue_motion=None,
        background_video=None,
        character_mask=None,
        frame_masks=None,
    ):
        """
        Execute the masked WAN animation.

        Args:
            frame_masks: Optional MASK tensor of shape (N, H, W) where N is the number of frames.
                        - 0.0 (black) = preserve original frame (no animation)
                        - 1.0 (white) = fully apply animation
                        - Values in between blend original and animated
        """
        trim_to_pose_video = False
        latent_length = ((length - 1) // 4) + 1
        latent_width = width // 8
        latent_height = height // 8
        trim_latent = 0

        if reference_image is None:
            reference_image = torch.zeros((1, height, width, 3))

        image = comfy.utils.common_upscale(
            reference_image[:length].movedim(-1, 1), width, height, "area", "center"
        ).movedim(1, -1)
        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = torch.zeros(
            (
                1,
                4,
                concat_latent_image.shape[-3],
                concat_latent_image.shape[-2],
                concat_latent_image.shape[-1],
            ),
            device=concat_latent_image.device,
            dtype=concat_latent_image.dtype,
        )
        trim_latent += concat_latent_image.shape[2]
        ref_motion_latent_length = 0

        if continue_motion is None:
            image = torch.ones((length, height, width, 3)) * 0.5
        else:
            continue_motion = continue_motion[-continue_motion_max_frames:]
            video_frame_offset -= continue_motion.shape[0]
            video_frame_offset = max(0, video_frame_offset)
            continue_motion = comfy.utils.common_upscale(
                continue_motion[-length:].movedim(-1, 1),
                width,
                height,
                "area",
                "center",
            ).movedim(1, -1)
            image = (
                torch.ones(
                    (length, height, width, continue_motion.shape[-1]),
                    device=continue_motion.device,
                    dtype=continue_motion.dtype,
                )
                * 0.5
            )
            image[: continue_motion.shape[0]] = continue_motion
            ref_motion_latent_length += ((continue_motion.shape[0] - 1) // 4) + 1

        if clip_vision_output is not None:
            positive = conditioning_set_values(
                positive, {"clip_vision_output": clip_vision_output}
            )
            negative = conditioning_set_values(
                negative, {"clip_vision_output": clip_vision_output}
            )

        if pose_video is not None:
            if pose_video.shape[0] <= video_frame_offset:
                pose_video = None
            else:
                pose_video = pose_video[video_frame_offset:]

        if pose_video is not None:
            pose_video = comfy.utils.common_upscale(
                pose_video[:length].movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
            if not trim_to_pose_video:
                if pose_video.shape[0] < length:
                    pose_video = torch.cat(
                        (pose_video,)
                        + (pose_video[-1:],) * (length - pose_video.shape[0]),
                        dim=0,
                    )

            pose_video_latent = vae.encode(pose_video[:, :, :, :3])
            positive = conditioning_set_values(
                positive, {"pose_video_latent": pose_video_latent}
            )
            negative = conditioning_set_values(
                negative, {"pose_video_latent": pose_video_latent}
            )

            if trim_to_pose_video:
                latent_length = pose_video_latent.shape[2]
                length = latent_length * 4 - 3
                image = image[:length]

        if face_video is not None:
            if face_video.shape[0] <= video_frame_offset:
                face_video = None
            else:
                face_video = face_video[video_frame_offset:]

        if face_video is not None:
            face_video = (
                comfy.utils.common_upscale(
                    face_video[:length].movedim(-1, 1), 512, 512, "area", "center"
                )
                * 2.0
                - 1.0
            )
            face_video = face_video.movedim(0, 1).unsqueeze(0)
            positive = conditioning_set_values(
                positive, {"face_video_pixels": face_video}
            )
            negative = conditioning_set_values(
                negative, {"face_video_pixels": face_video * 0.0 - 1.0}
            )

        ref_images_num = max(0, ref_motion_latent_length * 4 - 3)
        if background_video is not None:
            if background_video.shape[0] > video_frame_offset:
                background_video = background_video[video_frame_offset:]
                background_video = comfy.utils.common_upscale(
                    background_video[:length].movedim(-1, 1),
                    width,
                    height,
                    "area",
                    "center",
                ).movedim(1, -1)
                if background_video.shape[0] > ref_images_num:
                    image[ref_images_num : background_video.shape[0]] = (
                        background_video[ref_images_num:]
                    )

        # Create the base mask for reference motion
        mask_refmotion = torch.ones(
            (
                1,
                1,
                latent_length * 4,
                concat_latent_image.shape[-2],
                concat_latent_image.shape[-1],
            ),
            device=mask.device,
            dtype=mask.dtype,
        )
        if continue_motion is not None:
            mask_refmotion[:, :, : ref_motion_latent_length * 4] = 0.0

        # Apply character mask
        if character_mask is not None:
            if (
                character_mask.shape[0] > video_frame_offset
                or character_mask.shape[0] == 1
            ):
                if character_mask.shape[0] == 1:
                    character_mask = character_mask.repeat(
                        (length,) + (1,) * (character_mask.ndim - 1)
                    )
                else:
                    character_mask = character_mask[video_frame_offset:]
                if character_mask.ndim == 3:
                    character_mask = character_mask.unsqueeze(1)
                    character_mask = character_mask.movedim(0, 1)
                if character_mask.ndim == 4:
                    character_mask = character_mask.unsqueeze(1)
                character_mask = comfy.utils.common_upscale(
                    character_mask[:, :, :length],
                    concat_latent_image.shape[-1],
                    concat_latent_image.shape[-2],
                    "nearest-exact",
                    "center",
                )
                if character_mask.shape[2] > ref_images_num:
                    mask_refmotion[:, :, ref_images_num : character_mask.shape[2]] = (
                        character_mask[:, :, ref_images_num:]
                    )

        # Apply per-frame animation masks
        if frame_masks is not None:
            # frame_masks shape: (N, H, W) where N is number of frames
            # We need to process these masks to control which frames are animated

            # Handle frame offset
            if frame_masks.shape[0] > video_frame_offset or frame_masks.shape[0] == 1:
                if frame_masks.shape[0] == 1:
                    # Single mask - broadcast to all frames
                    frame_masks = frame_masks.repeat(
                        (length,) + (1,) * (frame_masks.ndim - 1)
                    )
                else:
                    # Multiple masks - apply offset
                    frame_masks = frame_masks[video_frame_offset:]

                # Ensure we have the right dimensions
                if frame_masks.ndim == 3:
                    frame_masks = frame_masks.unsqueeze(
                        1
                    )  # Add channel dim -> (N, 1, H, W)
                    frame_masks = frame_masks.movedim(0, 1)  # Move to (1, N, H, W)
                if frame_masks.ndim == 4:
                    frame_masks = frame_masks.unsqueeze(1)  # -> (1, 1, N, H, W)

                # Upscale to latent dimensions
                frame_masks = comfy.utils.common_upscale(
                    frame_masks[:, :, :length],
                    concat_latent_image.shape[-1],
                    concat_latent_image.shape[-2],
                    "bilinear",  # Use bilinear for smooth mask transitions
                    "center",
                )

                # Combine with existing mask_refmotion
                # frame_masks: 1.0 = animate, 0.0 = preserve original
                # mask_refmotion: current mask state
                # We multiply them together so both masks are respected
                if frame_masks.shape[2] > ref_images_num:
                    mask_refmotion[:, :, ref_images_num : frame_masks.shape[2]] *= (
                        frame_masks[:, :, ref_images_num:]
                    )

        concat_latent_image = torch.cat(
            (concat_latent_image, vae.encode(image[:, :, :, :3])), dim=2
        )

        # Transform mask to latent space format
        mask_refmotion = mask_refmotion.view(
            1,
            mask_refmotion.shape[2] // 4,
            4,
            mask_refmotion.shape[3],
            mask_refmotion.shape[4],
        ).transpose(1, 2)
        mask = torch.cat((mask, mask_refmotion), dim=2)

        positive = conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        latent = torch.zeros(
            [batch_size, 16, latent_length + trim_latent, latent_height, latent_width],
            device=comfy.model_management.intermediate_device(),
        )
        out_latent = {}
        out_latent["samples"] = latent

        return (
            positive,
            negative,
            out_latent,
            trim_latent,
            max(0, ref_motion_latent_length * 4 - 3),
            video_frame_offset + length,
        )


NODE_CLASS_MAPPINGS = {
    "WanAnimateToVideoMasked": WanAnimateToVideoMasked,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAnimateToVideoMasked": "WAN Animate to Video (Masked)",
}
