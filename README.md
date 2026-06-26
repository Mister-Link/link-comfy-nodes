# link-comfy-nodes

A collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) covering video editing, image processing, color grading, and workflow utilities.

## Nodes

**Video & Animation**
- **Average Mask Region Size** — average the white mask bounding-box size across frames
- **Video Mask Editor** — interactive mask painting over video frames with preview caching
- **VACE Sampler** — masked WAN/VACE frame repair with internal VACE conditioning
- **Stabilizer Trim** — trim stabilized video edges
- **Replace Alpha** — replace or inject alpha channels
- **Trim Conditioning** — trim VACE conditioning to match frame counts

**WAN Video Model**
- **WAN Frame Calculator** — compute frame counts for WAN's n×4+1 pattern
- **WAN Frames to Add & Cut** — calculate padding needed for target durations
- **Native WAN Pose Strength** — adjust pose control strength

**Image Processing**
- **Rotate Image** — batch rotation with configurable background fill
- **Pose Image Setup** — expand/offset images around a region
- **Crop to Content** — auto-crop to non-transparent content
- **Auto Cropper** — content-aware cropping with multiple detection methods (bbox, contour, alpha, anime segmentation)
- **Resize Image and Mask by Side** — resize by longest/shortest side
- **Add Image to Batch** — insert an image at a specific batch index
- **Pixelation Dimensions** — dimension presets
- **Remove Background** — bulk background removal via bgeraser.com
- **Spritesheet Builder** — combine frames into a spritesheet with target aspect ratio
- **Convert to Pixel Art** — pixelation effect with color quantization

**Color**
- **Convert Color Format** — parse hex/24-bit colors
- **Find Furthest Color** — find a color maximally distant from image pixels
- **Match Color Palette** — Reinhard Lab color grading between images

**Preview & Output**
- **Fast Image Preview** — thumbnail + full-res WebP previews
- **Preview Image (Alpha)** — preview RGBA images with transparency
- **Spritesheet Preview** — animated spritesheet preview
- **Preview (webm)** — encode and preview video as WebM

**Save**
- **Batch Image Save** — sequential image saving with auto-numbering
- **Save to ZIP** — save image/mask sequences into a ZIP archive
- **Save Folder as ZIP** — zip an existing output folder

**Text**
- **Concat** — template-based string concatenation (`%1`, `%2`, etc.)
- **Preview as Markdown** — render a string as markdown on the node

## Installation

Clone into your ComfyUI `custom_nodes` directory:

```
cd ComfyUI/custom_nodes
git clone https://github.com/linkoid/link-comfy-nodes.git
pip install -r link-comfy-nodes/requirements.txt
```

Restart ComfyUI. Nodes will appear under their respective categories.
