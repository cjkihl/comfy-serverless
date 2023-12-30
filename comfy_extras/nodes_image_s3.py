import base64
import io
from datetime import datetime
import os
import uuid
import boto3
from PIL import Image, ImageOps
import numpy as np
import torch


def create_image_id():
    date_string = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())
    return f"{date_string}-{unique_id}"


class SaveImageS3:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "bucket": ("STRING", {"default": "bucket-name"}),
                "prefix": ("STRING", {"default": "small"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def create_lqip(self, image: Image):
        i = image.resize((16, 16), Image.BICUBIC)
        buffer = io.BytesIO()
        i.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes)
        img_str = img_base64.decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def save_images(self, images, bucket, prefix="small"):
        s3 = boto3.client(
            service_name="s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            region_name="auto",  # Must be one of: wnam, enam, weur, eeur, apac, auto
        )
        results = []
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            image_id = create_image_id()
            key = f"{prefix}/{image_id}"

            # Convert the image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()

            # Upload image to S3 bucket
            s3.put_object(Bucket=bucket, Key=key, Body=img_byte_arr)
            results.append(
                {
                    "key": key,
                    "image_id": image_id,
                    "prefix": prefix,
                    "lqip": self.create_lqip(img),
                }
            )

        return {"ui": {"images": results}}


class LoadImageS3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bucket": ("STRING", {"default": "bucket-name"}),
                "key": ("STRING", {"default": "key"}),
            }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, bucket, key):
        s3 = boto3.client(
            service_name="s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            region_name="auto",  # Must be one of: wnam, enam, weur, eeur, apac, auto
        )
        response = s3.get_object(Bucket=bucket, Key=key)
        i = Image.open(io.BytesIO(response["Body"].read()))
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


NODE_CLASS_MAPPINGS = {"SaveImageS3": SaveImageS3, "LoadImageS3": LoadImageS3}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageS3": "Save Image S3",
    "LoadImageS3": "Load Image S3",
}
