"""Pose strength model patcher for native ComfyUI WAN Animate models."""


class NativeWanPoseStrength:
    """Model patcher that applies pose_strength scaling for WAN Animate models.

    This node patches the model's after_patch_embedding method to scale
    pose influence after the pose_patch_embedding convolution, which is the
    mathematically correct place to apply strength (matching WanVideoWrapper's approach).

    Connect this between your model loader and the sampler.
    """

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "conditioning/video_models"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "pose_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "overdrive": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    def patch_model(self, model, pose_strength: float, overdrive: float):
        pose_strength = float(pose_strength)
        overdrive = float(overdrive)

        if pose_strength >= 1.0:
            effective_strength = 2.5 + (2.0 * max(0.0, min(1.0, overdrive)))
        else:
            effective_strength = pose_strength * 2.5

        if abs(effective_strength - 1.0) < 1e-6:
            # No patching needed for default strength
            return (model,)

        # Clone the model to avoid modifying the original
        patched_model = model.clone()

        # Get the diffusion model
        diffusion_model = patched_model.model.diffusion_model

        # Check if this is a WAN Animate model with pose_patch_embedding
        if not hasattr(diffusion_model, "pose_patch_embedding"):
            # Not a WAN Animate model, return unchanged
            return (patched_model,)

        # Store original method
        original_after_patch_embedding = diffusion_model.after_patch_embedding

        # Create patched method that applies strength scaling
        def patched_after_patch_embedding(x, pose_latents, face_pixel_values):
            if pose_latents is not None:
                # Apply pose_patch_embedding (Conv3d)
                pose_latents_embedded = diffusion_model.pose_patch_embedding(
                    pose_latents
                )
                # Apply strength scaling AFTER the convolution (correct approach)
                # Then add to x with the strength factor
                x[:, :, 1 : pose_latents_embedded.shape[2] + 1] += (
                    pose_latents_embedded[:, :, : x.shape[2] - 1] * effective_strength
                )
                # Now handle face_pixel_values using original logic but skip pose part
                # We need to call original but prevent it from processing pose again
                _, motion_vec = original_after_patch_embedding(
                    x, None, face_pixel_values
                )
                return x, motion_vec
            else:
                # No pose latents, use original method
                return original_after_patch_embedding(
                    x, pose_latents, face_pixel_values
                )

        # Apply the patch
        diffusion_model.after_patch_embedding = patched_after_patch_embedding

        return (patched_model,)
