import io
import os
from typing import Optional
import boto3
from botocore.exceptions import BotoCoreError

from PIL import Image, ImageOps
import numpy as np
import torch
import time
from .face import FaceData, face_data_to_dict
from .images import create_image_id, pil_to_bytes, tensor_to_pil, create_lqip


class SaveImageS3:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "bucket": ("STRING", {"default": "bucket-name"}),
                "prefix": ("STRING", {"default": ""}),
            },
            "optional": {
                "segs": ("SEGS",),
                "face_data": ("FACEDATA",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "CJ Nodes"

    def save_images(
        self,
        images: torch.Tensor,
        bucket: str,
        prefix: str = "",
        segs: Optional[list[dict]] = None,
        face_data: Optional[list[list[FaceData]]] = None,
    ) -> dict:
        """Upload images to S3 bucket with LQIP and face data.

        Args:
            images: Tensor of images to upload
            bucket: S3 bucket name
            prefix: Optional prefix for S3 keys
            segs: Optional segmentation data
            face_data: Optional face detection data
        """
        url = os.getenv("S3_ENDPOINT_URL")
        if url is None:
            raise ValueError("Environment variable S3_ENDPOINT_URL is not set")

        try:
            s3 = boto3.client(
                service_name="s3",
                endpoint_url=url,
                region_name="auto",
            )

            to_upload = []
            for image in images:
                img = tensor_to_pil(image)
                image_id = create_image_id()
                lqip = create_lqip(img)
                to_upload.append(
                    {
                        "bytes": pil_to_bytes(img, "PNG", 100),
                        "lqip": lqip,
                        "id": image_id,
                    }
                )

            results = []
            for i, image in enumerate(to_upload):
                key = f"{prefix}/{image['id']}" if prefix else image["id"]

                try:
                    start_time = time.time()
                    s3.put_object(Bucket=bucket, Key=key, Body=image["bytes"])
                    end_time = time.time()
                    print(
                        f"Time taken to upload image: {end_time - start_time} seconds"
                    )

                    f = []
                    if face_data is not None and i < len(face_data):
                        for face in face_data[i]:
                            f.append(face_data_to_dict(face))

                    results.append(
                        {
                            "key": key,
                            "image_id": image["id"],
                            "prefix": prefix,
                            "segs": segs[i] if segs else [],
                            "face_data": f,
                            "lqip": image["lqip"],
                        }
                    )

                except BotoCoreError as e:
                    print(f"Failed to upload image {key}: {str(e)}")
                    continue

            return {"ui": {"result": results}}

        except Exception as e:
            raise RuntimeError(f"S3 upload failed: {str(e)}")


class LoadImageS3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bucket": ("STRING", {"default": "bucket-name"}),
                "key": ("STRING", {"default": "key"}),
            }
        }

    CATEGORY = "CJ Nodes"

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
