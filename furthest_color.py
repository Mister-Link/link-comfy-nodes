import torch
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import differential_evolution

class FarthestColorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "sample_rate": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("24-bit", "hex", "rgb")
    FUNCTION = "find_farthest_color"
    CATEGORY = "utils"
    
    def find_farthest_color(self, images, sample_rate=10):
        """
        Find the color in RGB space that maximizes the minimum distance to all colors 
        present in the image batch. This finds the optimal greenscreen color.
        
        Args:
            images: Batch of images in ComfyUI format (B, H, W, C) with values in [0, 1]
            sample_rate: Sample every Nth pixel to speed up computation (default: 10)
        """
        # Convert images to numpy and extract RGB values
        images_np = images.cpu().numpy()
        
        # Sample pixels to reduce computation
        sampled_pixels = images_np[:, ::sample_rate, ::sample_rate, :3]
        
        # Reshape to (num_pixels, 3) and convert to 0-255 range
        pixels = sampled_pixels.reshape(-1, 3) * 255.0
        pixels = pixels.astype(np.float32)
        
        # Remove duplicates and build KDTree for fast nearest neighbor queries
        unique_pixels = np.unique(pixels, axis=0)
        print(f"Found {len(unique_pixels)} unique colors in image")
        tree = KDTree(unique_pixels)
        
        # Objective function: maximize the minimum distance to any color in the image
        # We negate because scipy minimizes
        def objective(color):
            distance, _ = tree.query(color)
            return -distance  # Negate to maximize
        
        # Use differential evolution for global optimization
        # This explores the entire RGB space intelligently
        bounds = [(0, 255), (0, 255), (0, 255)]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            atol=0.01,
            tol=0.01
        )
        
        best_color = result.x
        
        # Round to integer RGB values
        r, g, b = np.round(best_color).astype(int)
        r, g, b = np.clip([r, g, b], 0, 255)
        
        # Verify this color doesn't exist in the image
        test_color = np.array([r, g, b], dtype=np.float32)
        distance_to_nearest, _ = tree.query(test_color)
        
        print(f"Selected color: RGB({r}, {g}, {b})")
        print(f"Distance to nearest image color: {distance_to_nearest:.2f}")
        
        # If we somehow landed very close to an existing color, adjust slightly
        if distance_to_nearest < 1.0:
            # Find the nearest color and move away from it
            _, idx = tree.query(test_color)
            nearest = unique_pixels[idx]
            
            # Move in opposite direction
            direction = test_color - nearest
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                adjusted = test_color + 2.0 * direction
                r, g, b = np.round(np.clip(adjusted, 0, 255)).astype(int)
                print(f"Adjusted to: RGB({r}, {g}, {b})")
        
        # Convert to 24-bit integer
        value_int = (int(r) << 16) | (int(g) << 8) | int(b)
        
        # Format outputs
        hex_str = f"#{value_int:06X}"
        rgb_str = f"{r}, {g}, {b}"
        
        return (int(value_int), hex_str, rgb_str)

NODE_CLASS_MAPPINGS = {
    "Farthest Color": FarthestColorNode,
}
