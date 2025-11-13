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
                "max_brightness": ("INT", {
                    "default": 140,
                    "min": 50,
                    "max": 200,
                    "step": 5,
                    "display": "number"
                }),
                "min_saturation": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "display": "number"
                }),
                "max_saturation": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.3,
                    "max": 0.9,
                    "step": 0.05,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("24-bit", "hex", "rgb")
    FUNCTION = "find_farthest_color"
    CATEGORY = "utils"
    
    def rgb_to_hsv(self, r, g, b):
        """Convert RGB (0-255) to HSV (H: 0-360, S: 0-1, V: 0-1)"""
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        diff = max_c - min_c
        
        if max_c == min_c:
            h = 0
        elif max_c == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_c == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360
        
        s = 0 if max_c == 0 else diff / max_c
        v = max_c
        
        return h, s, v
    
    def is_valid_color(self, r, g, b, max_brightness, min_sat, max_sat):
        """Check if color meets the 'darker, muted but not gray' criteria"""
        h, s, v = self.rgb_to_hsv(r, g, b)
        
        # Value (brightness) should be darker
        brightness = v * 255
        if brightness > max_brightness:
            return False
        
        # Saturation should be moderate (not gray, not neon)
        if s < min_sat or s > max_sat:
            return False
        
        return True
    
    def find_farthest_color(self, images, sample_rate=10, max_brightness=140, 
                           min_saturation=0.2, max_saturation=0.6):
        """
        Find the color that maximizes minimum distance to image colors,
        constrained to darker, muted tones suitable for video generation.
        
        Args:
            images: Batch of images in ComfyUI format (B, H, W, C) with values in [0, 1]
            sample_rate: Sample every Nth pixel (default: 10)
            max_brightness: Maximum brightness/value (default: 140, range: 0-255)
            min_saturation: Minimum saturation to avoid gray (default: 0.2)
            max_saturation: Maximum saturation to avoid neon (default: 0.6)
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
        
        # Objective function with constraints
        def objective(color):
            r, g, b = color
            
            # Check constraints
            if not self.is_valid_color(r, g, b, max_brightness, min_saturation, max_saturation):
                return 1e6  # High penalty for invalid colors
            
            # Find distance to nearest image color
            distance, _ = tree.query(color)
            return -distance  # Negate to maximize
        
        # Constrain search space based on max_brightness
        bounds = [
            (0, max_brightness),
            (0, max_brightness),
            (0, max_brightness)
        ]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=150,
            popsize=20,
            seed=42,
            atol=0.01,
            tol=0.01
        )
        
        best_color = result.x
        
        # Round to integer RGB values
        r, g, b = np.round(best_color).astype(int)
        r, g, b = np.clip([r, g, b], 0, 255)
        
        # Verify the color meets our constraints
        if not self.is_valid_color(r, g, b, max_brightness, min_saturation, max_saturation):
            print(f"Warning: Optimized color RGB({r}, {g}, {b}) doesn't meet constraints, adjusting...")
            # Fall back to a safe dark muted color
            r, g, b = 80, 100, 90  # Dark muted teal-ish
        
        # Verify this color doesn't exist in the image
        test_color = np.array([r, g, b], dtype=np.float32)
        distance_to_nearest, _ = tree.query(test_color)
        
        h, s, v = self.rgb_to_hsv(r, g, b)
        print(f"Selected color: RGB({r}, {g}, {b}) / HSV({h:.1f}°, {s:.2f}, {v:.2f})")
        print(f"Distance to nearest image color: {distance_to_nearest:.2f}")
        
        # Convert to 24-bit integer
        value_int = (int(r) << 16) | (int(g) << 8) | int(b)
        
        # Format outputs
        hex_str = f"#{value_int:06X}"
        rgb_str = f"{r}, {g}, {b}"
        
        return (int(value_int), hex_str, rgb_str)

NODE_CLASS_MAPPINGS = {
    "Farthest Color": FarthestColorNode,
}
