import io
import os
from typing import Optional
import boto3
from PIL import Image, ImageOps
import numpy as np
import torch
import time
from utils.face import FaceData
from utils.images import create_image_id, pil_to_bytes, tensor_to_pil, create_lqip


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
        images,
        bucket,
        prefix="",
        segs=None,
        face_data: Optional[list[list[FaceData]]] = None,
    ):
        url = os.getenv("S3_ENDPOINT_URL")
        if url is None:
            raise ValueError("Environment variable S3_ENDPOINT_URL is not set")
        s3 = boto3.client(
            service_name="s3",
            endpoint_url=url,
            region_name="auto",  # Must be one of: wnam, enam, weur, eeur, apac, auto
        )

        to_upload = []
        lqip_list = []
        for image in images:
            img = tensor_to_pil(image)
            image_id = create_image_id()
            lqip = create_lqip(img)
            to_upload.append(
                {"bytes": pil_to_bytes(img, "PNG", 100), "lqip": lqip, "id": image_id}
            )
            lqip_list.append(lqip)

        results = []

        for i, image in enumerate(to_upload):
            key = f"{prefix}/{image['id']}" if prefix else image["id"]

            start_time = time.time()
            # Upload image to S3 bucket
            s3.put_object(Bucket=bucket, Key=key, Body=image["bytes"])
            end_time = time.time()
            print(f"Time taken to upload image: {end_time - start_time} seconds")

            f = []
            if face_data is not None:
                for face in face_data[i]:
                    b = {}
                    b["landmarks"] = face.landmarks
                    b["bbox"] = face.bbox
                    f.append(b)

            results.append(
                {
                    "key": key,
                    "image_id": image["id"],
                    "prefix": prefix,
                    "segs": [],
                    "face_data": f,
                    "lqip": image["lqip"],
                }
            )

        return {"ui": {"result": results}}


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
