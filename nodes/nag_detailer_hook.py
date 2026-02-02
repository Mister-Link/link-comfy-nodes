"""Detailer hook for NAG compatibility - ensures dimensions stay divisible during upscaling."""

from __future__ import annotations


class NAGCompatibleDetailerHook:
    """
    Detailer hook that ensures upscaled dimensions remain divisible by a specified value.

    This is critical for NAG (Normalized Attention Guidance) models which require
    dimensions divisible by 64 (8 for VAE × 8 for attention heads) to avoid
    "Unexpected floating ScalarType in at::autocast::prioritize" errors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "divisor": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("DETAILER_HOOK",)
    FUNCTION = "create_hook"
    CATEGORY = "link/Impact"
    DESCRIPTION = "Creates a detailer hook that ensures upscaled dimensions are NAG-compatible (divisible by 64)"

    def create_hook(self, divisor: int = 64):
        """Create a hook that rounds dimensions to nearest multiple of divisor."""
        hook = NAGDetailerHookImpl(divisor)
        return (hook,)


class NAGDetailerHookImpl:
    """Implementation of the NAG-compatible detailer hook."""

    def __init__(self, divisor: int = 64):
        self.divisor = divisor

    def touch_scaled_size(self, width: int, height: int) -> tuple[int, int]:
        """
        Called by DetailerForAnimateDiff to adjust upscaled dimensions.

        Rounds dimensions down to nearest multiple of divisor to ensure
        NAG compatibility during sampling.
        """
        # Round down to nearest multiple of divisor
        adjusted_width = (width // self.divisor) * self.divisor
        adjusted_height = (height // self.divisor) * self.divisor

        # Ensure minimum size
        if adjusted_width < self.divisor:
            adjusted_width = self.divisor
        if adjusted_height < self.divisor:
            adjusted_height = self.divisor

        if adjusted_width != width or adjusted_height != height:
            print(
                f"[NAG Hook] Adjusted upscaled size from ({width}, {height}) "
                f"to ({adjusted_width}, {adjusted_height}) (divisor={self.divisor})"
            )

        return adjusted_width, adjusted_height

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

    def get_custom_sampler(self):
        """Return custom sampler if needed - None for default."""
        return None


__all__ = ["NAGCompatibleDetailerHook"]
