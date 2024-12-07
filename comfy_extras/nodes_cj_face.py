import os
from comfy_execution.graph import ExecutionBlocker
import torch
from insightface.app import FaceAnalysis
from insightface.app.common import Face
import folder_paths
from PIL import Image
import numpy as np
import torchvision.transforms.v2 as T
import cv2
from utils.face import (
    FaceData,
    calculate_square_bounds,
    dilate_mask,
    feather_mask,
    resize_face_image,
)

INSIGHTFACE_MODELS_DIR = os.path.join(folder_paths.models_dir, "insightface")


def get_folder_names(directory: str):
    """Get list of folder names from directory"""
    try:
        # Get all items and filter for directories
        folder_names = [
            item
            for item in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, item))
        ]
        return sorted(folder_names)
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return []


class INSIGHFACE_MODEL_LOADER:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM", "CoreML"],),
                "name": (
                    get_folder_names(os.path.join(INSIGHTFACE_MODELS_DIR, "models")),
                ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS", "INSIGHTFACE")
    FUNCTION = "load_insight_face"
    CATEGORY = "CJ Face Nodes"

    def load_insight_face(self, provider, name):

        print("Loading InsightFace model")
        print("Model Name: ", name)
        model = FaceAnalysis(
            name=name,
            root=INSIGHTFACE_MODELS_DIR,
            providers=[
                provider + "ExecutionProvider",
            ],
        )
        model.prepare(ctx_id=0, det_size=(640, 640))

        # Return the model as 2 dfferent types to support different nodes
        return (model, model)


def tensor_to_image(image: torch.Tensor) -> Image.Image:
    return T.ToPILImage()(image.permute(2, 0, 1)).convert("RGB")


class GET_FACES:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "insightface": ("INSIGHTFACE",),
            },
            "optional": {
                "max_num": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 1}),
                "min_face_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("FACEDATA",)
    RETURN_NAMES = ("face_data",)
    FUNCTION = "get_faces"
    CATEGORY = "CJ Face Nodes"

    def get_faces(
        self,
        images: torch.Tensor,
        insightface: FaceAnalysis,
        max_num: int = 10,
        min_face_size: int = 0,
    ):
        batch_face_data: list[list[FaceData]] = [[] for _ in range(len(images))]

        if max_num == 0:
            return (batch_face_data,)

        for i, img in enumerate(images):
            # Get faces for single image
            faces: list[Face] = insightface.get(np.array(tensor_to_image(img)), max_num)
            if faces is None:
                batch_face_data.append([])
                continue

            # Create and sort face data
            face_data = [
                FaceData(face["bbox"], face["landmark_2d_106"]) for face in faces
            ]

            # Filter faces by minimum size if needed
            face_data = (
                [
                    face
                    for face in face_data
                    if face.bbox[2] - face.bbox[0] >= min_face_size
                ]
                if min_face_size > 0
                else face_data
            )

            if (len(face_data)) == 0:
                batch_face_data.append([])
                continue

            face_data.sort(
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )
            batch_face_data[i] = face_data

        return (batch_face_data,)


class HAS_FACES:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_data": ("FACEDATA",),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FACEDATA", "IMAGE", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("face data", "images bypass", "has_faces", "no_faces")
    FUNCTION = "has_faces"
    CATEGORY = "CJ Face Nodes"

    def has_faces(
        self,
        images: torch.Tensor,
        face_data: list[list[FaceData]],
    ):
        # Check if there are faces in the face_data
        face_count = sum(len(faces) for faces in face_data)
        if face_count > 0:
            return (face_data, ExecutionBlocker(None), True, False)
        else:
            return (ExecutionBlocker(None), images, False, True)


LANDMARK_REGIONS = [
    "eyes",
    "left_brow",
    "left_eye",
    "main_features",
    "mouth",
    "nose",
    "outline",
    "outline_forehead",
    "right_brow",
    "right_eye",
]


class FACE_DETAILER_CROP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "face_data": ("FACEDATA",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "expand": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "landmarks": (LANDMARK_REGIONS, {"default": "main_features"}),
            },
            "optional": {
                "face_size": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "FACECROPDATA")
    RETURN_NAMES = ("face_masks", "face_images", "face_crop_data")
    FUNCTION = "get_face_masks"
    CATEGORY = "CJ Face Nodes"

    # Constants

    MIN_HULL_POINTS = 3

    def get_face_masks(
        self,
        images: torch.Tensor,  # (B, H, W, C)
        face_data: list[list[FaceData]],  # List of list of FaceData
        padding: int,  # Padding percent around face
        feather: int,  # Feathering percent around face
        expand: int,  # Percent to expand mask
        landmarks: str,
        face_size: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int, int, int]]]:

        # Early validation
        batch_size, images_height, images_width, _ = images.shape
        if batch_size != len(face_data):
            raise ValueError("Number of images must match number of face lists")

        # Pre-calculate total faces for tensor allocation
        total_faces = sum(len(faces) for faces in face_data)

        # Initialize output tensors
        face_masks = torch.zeros((total_faces, face_size, face_size))
        face_images = torch.zeros((total_faces, face_size, face_size, 3))
        face_crop_data: list[tuple[int, int, int, int, int]] = []

        # Pre-allocate reusable arrays
        face_mask = np.zeros((face_size, face_size), dtype=np.float32)

        face_index = 0
        for batch_idx, faces in enumerate(face_data):
            for face in faces:
                # Skip invalid faces
                if (
                    landmarks not in face.landmarks
                    or len(face.landmarks[landmarks]) == 0
                ):
                    continue

                hull_points = cv2.convexHull(face.landmarks[landmarks])
                if hull_points is None or len(hull_points) < self.MIN_HULL_POINTS:
                    continue

                x, y, w, h = cv2.boundingRect(hull_points)

                # Get square bounding box
                x1, y1, x2, y2 = calculate_square_bounds(
                    w,
                    h,
                    x + w // 2,
                    y + h // 2,
                    padding,
                    images_width,
                    images_height,
                )

                # Create mask
                face_mask.fill(0)
                # Convert hull points to face size
                roi_points = (hull_points - np.array([[x1, y1]])) * (
                    face_size / (x2 - x1)
                )
                cv2.fillConvexPoly(face_mask, roi_points.astype(np.int32), (1, 1, 1))

                # If expand dilate the mask.
                if expand > 0:
                    face_mask = dilate_mask(face_mask, expand)

                # Apply feathering
                if feather > 0:
                    face_mask = feather_mask(face_mask, feather)

                # Store results
                face_masks[face_index] = torch.from_numpy(face_mask)

                # Crop and resize face image
                face_image = images[batch_idx, y1:y2, x1:x2]  # [H, W, C]
                face_image = resize_face_image(face_image, face_size)
                print(f"After resize - range: [{face_image.min()}, {face_image.max()}]")

                # Store results
                face_images[face_index] = face_image

                face_crop_data.append((batch_idx, x1, y1, x2, y2))
                face_index += 1

        return face_masks[:face_index], face_images[:face_index], face_crop_data


class FACE_DETAILER_STITCH:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "face_images": ("IMAGE",),
                "face_crop_data": ("FACECROPDATA",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "stitch_face_images"
    CATEGORY = "CJ Face Nodes"

    def stitch_face_images(
        self,
        images: torch.Tensor,  # (B, H, W, C) The original images to stitch faces onto
        face_images: torch.Tensor,  # (B, H, W, C) The face images to stitch
        face_crop_data: list[
            tuple[int, int, int, int, int]
        ],  # List face crop data (batch_idx, x1, y1, x2, y2)
    ):

        if len(face_crop_data) != len(face_images):
            raise ValueError(
                "Number of face crop data must match number of face images"
            )

        # Create a copy of input images
        result_images = images.clone()

        face_index = 0
        for batch_idx, x1, y1, x2, y2 in face_crop_data:
            face_image = face_images[face_index].clone()
            face_index += 1
            face_image = resize_face_image(face_image, x2 - x1)
            # Copy the face image onto the original image
            result_images[batch_idx, y1:y2, x1:x2, :] = face_image
        return (result_images,)


NODE_CLASS_MAPPINGS = {
    "InsightFaceModelLoader": INSIGHFACE_MODEL_LOADER,
    "GetFaces": GET_FACES,
    "HasFaces": HAS_FACES,
    "FaceDetailerCrop": FACE_DETAILER_CROP,
    "FaceDetailerStitch": FACE_DETAILER_STITCH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InsightFaceModelLoader": "InsightFace Model Loader",
    "GetFaces": "Get Faces",
    "HasFaces": "Has Faces",
    "FaceDetailerCrop": "Face Detailer Crop #1",
    "FaceDetailerStitch": "Face Detailer Stitch #2",
}


# The tensor image will have the format (B, H, W, C), where:
# B is the batch dimension.
# H is the height of the image.
# W is the width of the image.
# C is the number of channels (3 for RGB).
# The values in the tensor will be in the range [0, 1].
