from __future__ import annotations


class AddImageToBatchNode:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "add_image"
    CATEGORY = "image/batch"
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "batch": ("IMAGE",),
                "index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 999, "step": 1},
                ),
            }
        }

    def add_image(self, image: list, batch: list, index: list):
        img = image[0].detach().float()

        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.ndim != 4:
            raise ValueError("Expected image with shape (N, H, W, C)")

        single_image = img[0:1]

        idx = index[0] if isinstance(index[0], int) else int(index[0].item())
        idx = max(0, idx)

        batch_images = []
        for bat_tensor in batch:
            bat = bat_tensor.detach().float()
            if bat.ndim == 3:
                bat = bat.unsqueeze(0)
            if bat.ndim == 4:
                for i in range(bat.shape[0]):
                    batch_images.append(bat[i : i + 1])

        idx = min(idx, len(batch_images))

        result_list = batch_images[:idx] + [single_image] + batch_images[idx:]

        return (result_list,)
