import base64
import io
import os
from PIL import Image, ImageOps
import numpy as np
import torch
import folder_paths


def load_base64_image(base64_str):
    img_bytes = base64.b64decode(base64_str)
    i = Image.open(io.BytesIO(img_bytes))
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


@torch.no_grad()
def save_base64_image(images: torch.Tensor):
    for image in images:
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        prefix = "data:image/jpeg;base64,"
        return img_str.decode("utf-8")


@torch.no_grad()
def save_image(images: torch.Tensor, filename_prefix: str = "ComfyUI"):
    """Save Images"""
    filename_prefix += ""
    (
        full_output_folder,
        filename,
        counter,
        subfolder,
        filename_prefix,
    ) = folder_paths.get_save_image_path(
        filename_prefix,
        folder_paths.get_output_directory(),
        images[0].shape[1],
        images[0].shape[0],
    )
    results = []
    for image in images:
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = None

        file = f"{filename}_{counter:05}_.png"
        img.save(
            os.path.join(full_output_folder, file),
            pnginfo=metadata,
            compress_level=4,
        )
        results.append({"filename": file, "subfolder": subfolder, "type": "output"})
        counter += 1

    return results
