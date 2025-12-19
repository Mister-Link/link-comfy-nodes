import numpy as np
import torch
import torch.nn as nn
from models.module_edge_detector import EdgeDetectorModule
from models.module_pixel_effect import PixelEffectModule
from PIL import Image


class Photo2PixelModel(nn.Module):
    def __init__(self):
        super(Photo2PixelModel, self).__init__()
        self.module_pixel_effect = PixelEffectModule()
        self.module_edge_detect = EdgeDetectorModule()

    def forward(
        self, rgb, param_kernel_size=10, param_pixel_size=16, param_edge_thresh=112
    ):
        """
        Process RGB image with pixelation and edge detection.

        :param rgb: [b(1), c(3), H, W] - RGB image tensor
        :param param_kernel_size: Kernel size for pixelation
        :param param_pixel_size: Pixel size for pixelation
        :param param_edge_thresh: Edge detection threshold (0-255)
        :return: [b(1), c(3), H, W] - Processed RGB image
        """
        rgb = self.module_pixel_effect(rgb, 4, param_kernel_size, param_pixel_size)

        edge_mask = self.module_edge_detect(rgb, param_edge_thresh, param_edge_dilate=3)
        rgb = torch.masked_fill(rgb, torch.gt(edge_mask, 0.5), 0)

        return rgb


def test1():
    img = Image.open("../images/example_input_mountain.jpg").convert("RGB")
    img_np = np.array(img).astype(np.float32)
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]
    img_pt = torch.from_numpy(img_np)

    model = Photo2PixelModel()
    model.eval()

    with torch.no_grad():
        result_rgb_pt = model(img_pt, param_kernel_size=11, param_pixel_size=16)
        result_rgb_pt = result_rgb_pt[0, ...].permute(1, 2, 0)

    print("img_pt", img_pt.shape)
    print("result_rgb_pt", result_rgb_pt.shape)

    result_rgb_np = result_rgb_pt.cpu().numpy().astype(np.uint8)
    Image.fromarray(result_rgb_np).save("./test_result_photo2pixel.png")


if __name__ == "__main__":
    test1()
