import base64
import io
from PIL import Image, ImageOps
import numpy as np
import torch


DEFAULT_MIME_TYPE = "WEBP"
LQIP_SIZE = 16
IMAGE_QUALITY = 100


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor [H,W,C] to PIL Image"""
    i = 255.0 * tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img


def pil_to_base64(
    image: Image.Image, mime_type: str = "JPEG", quality: int = 100
) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type, quality=quality)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


class SaveLqip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "lqip_size": ("INT", {"default": LQIP_SIZE, "min": 1}),
                "mime_type": ("STRING", {"default": DEFAULT_MIME_TYPE}),
                "quality": ("INT", {"default": 20, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image"

    OUTPUT_NODE = True

    CATEGORY = "CJ Nodes"

    def save_image(
        self, images: torch.Tensor, lqip_size: int, mime_type: str, quality: int
    ):
        """Create low quality image placeholders from tensor batch"""
        results = []
        for image in images:
            # Convert tensor to PIL
            pil = tensor_to_pil(image)

            # Calculate aspect ratio
            aspect_ratio = pil.width / pil.height

            # Resize maintaining aspect ratio
            pil = pil.resize(
                (lqip_size, round(lqip_size / aspect_ratio)), Image.BICUBIC
            )

            # Convert to base64
            base64_str = pil_to_base64(pil, mime_type, quality)
            results.append({"image": base64_str})

        return {"ui": {"result": results}}


class SaveImageBase64:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mime_type": ("STRING", {"default": DEFAULT_MIME_TYPE}),
                "quality": ("INT", {"default": IMAGE_QUALITY, "min": 10, "max": 100}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image"

    OUTPUT_NODE = True

    CATEGORY = "CJ Nodes"

    def save_image(self, images: torch.Tensor, mime_type: str, quality: int):
        """Create low quality image placeholders from tensor batch"""
        results = []
        for image in images:
            # Convert tensor to PIL
            pil = tensor_to_pil(image)

            # Convert to base64
            base64_str = pil_to_base64(pil, mime_type, quality)
            results.append({"image": base64_str})

        return {"ui": {"result": results}}


class LoadImageBase64:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_base64": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "CJ Nodes"

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
    "SaveLqip": SaveLqip,
    "SaveImageBase64": SaveImageBase64,
    "LoadImageBase64": LoadImageBase64,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveLqip": "Save LQIP",
    "SaveImageBase64": "Save Image Base64",
    "LoadImageBase64": "Load Image Base64",
}
