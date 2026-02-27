"""Update conditioning to match a new latent after crop/resize."""

from __future__ import annotations


# Conditioning keys that encode spatial tensors sized to the previous
# latent dimensions. These must be dropped when the latent is resized so
# the next sampler doesn't receive mismatched spatial conditioning.
_SPATIAL_KEYS = frozenset({
    "concat_latent_image",
    "concat_mask",
    "vace_frames",
    "vace_mask",
    "pose_video_latent",
    "control_video",
    "reference_latents",
    "time_dim_concat",
})


class ChangeLatentDimensions:
    """Drop spatially-bound conditioning keys after a latent crop/resize.

    Use this between stages when you VAE-decode, crop the frames with Auto
    Cropper, then VAE-encode the cropped frames back to a smaller latent.
    The node passes the new latent through unchanged and strips any
    conditioning entries whose spatial tensors were sized for the old
    dimensions (concat_latent_image, vace_frames, vace_mask, etc.).

    Conditioning that is not spatially bound (text embeddings, clip_vision,
    vace_strength, audio embeds, etc.) is preserved as-is.
    """

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "positive", "negative")
    FUNCTION = "execute"
    CATEGORY = "latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            }
        }

    @staticmethod
    def _strip_spatial(cond: list) -> list:
        out = []
        for tensor, meta in cond:
            new_meta = {k: v for k, v in meta.items() if k not in _SPATIAL_KEYS}
            out.append((tensor, new_meta))
        return out

    def execute(self, latent, positive, negative):
        pos = self._strip_spatial(positive)
        neg = self._strip_spatial(negative)
        return (latent, pos, neg)
