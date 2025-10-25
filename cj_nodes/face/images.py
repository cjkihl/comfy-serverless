import base64
import io
from PIL import Image
import numpy as np
import torch
from datetime import datetime
import cuid


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


def pil_to_bytes(
    image: Image.Image, mime_type: str = "JPEG", quality: int = 100
) -> bytes:
    """Convert PIL Image to bytes"""
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type, quality=quality)
    return buffer.getvalue()


def create_lqip(
    image: Image.Image, lqip_size: int = 16, mime_type="WEBP", quality=20
) -> str:
    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    image = image.resize((lqip_size, round(lqip_size / aspect_ratio)), Image.BICUBIC)
    # Convert to base64
    return pil_to_base64(image, mime_type, quality)


def create_image_id():
    date_string = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(cuid.cuid())
    return f"{date_string}-{unique_id}"

