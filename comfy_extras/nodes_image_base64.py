import base64
import io
from PIL import Image, ImageOps
import numpy as np
import torch


class SaveImageBase64:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "mime_type": ("STRING", {"default": "WEBP"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def img_to_base64(self, image: Image, mime_type: str = "WEBP"):
        buffer = io.BytesIO()
        image.save(buffer, format=mime_type, quality=20)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes)
        return img_base64.decode("utf-8")

    def save_images(self, images, mime_type: str = "WEBP"):
        results = []
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            results.append(self.img_to_base64(img, mime_type=mime_type))

        return {"ui": {"result": results}}


class LoadImageBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_base64": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image_base64: str):
        image_bytes = base64.b64decode(image_base64)
        buffer = io.BytesIO(image_bytes)
        i = Image.open(buffer)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (image, mask.unsqueeze(0))


NODE_CLASS_MAPPINGS = {
    "SaveImageBase64": SaveImageBase64,
    "LoadImageBase64": LoadImageBase64,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageBase64": "Save Image Base64",
    "LoadImageBase64": "Load Image Base64",
}
