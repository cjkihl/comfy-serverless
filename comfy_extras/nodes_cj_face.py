import os
import torch
from insightface.app import FaceAnalysis
from insightface.app.common import Face
import folder_paths
from PIL import Image
import numpy as np
from numpy.typing import NDArray
import torchvision.transforms.v2 as T
import cv2


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


# PyTorch tensor format is (B,H,W,C)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return T.ToTensor()(image).permute(1, 2, 0)


class FaceData:
    def __init__(self, bbox: np.ndarray, landmarks: np.ndarray):
        self.bbox: tuple[int, int, int, int] = tuple(bbox.astype(int))
        landmarks = landmarks.astype(np.int64)
        self.landmarks: dict[str, NDArray[np.int64]] = {
            "main_features": landmarks[33:],
            "left_eye": landmarks[87:97],
            "right_eye": landmarks[33:43],
            "eyes": landmarks[[*range(33, 43), *range(87, 97)]],
            "nose": landmarks[72:87],
            "mouth": landmarks[52:72],
            "left_brow": landmarks[97:106],
            "right_brow": landmarks[43:52],
            "outline": landmarks[[*range(33), *range(48, 51), *range(102, 105)]],
            "outline_forehead": landmarks[
                [*range(33), *range(48, 51), *range(102, 105)]
            ],
        }


class GET_FACES:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "insightface": ("INSIGHTFACE",),
            },
            "optional": {
                "max_num": ("INT", {"default": 10, "min": 1, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("FACEDATA",)
    RETURN_NAMES = ("face_data",)
    FUNCTION = "get_faces"
    CATEGORY = "CJ Face Nodes"

    def get_faces(self, images: np.ndarray, insightface: FaceAnalysis, max_num=10):
        batch_face_data = []
        for img in images:
            # Get faces for single image
            faces: list[Face] = insightface.get(np.array(tensor_to_image(img)), max_num)
            if faces is None:
                batch_face_data.append([])
                continue

            # Create and sort face data
            face_data = [
                FaceData(face["bbox"], face["landmark_2d_106"]) for face in faces
            ]
            face_data.sort(
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )
            batch_face_data.append(face_data)

        return (batch_face_data,)


def expand_bbox(
    w: int,
    h: int,
    bbox: tuple[int, int, int, int],
    padding: int = 0,
    padding_percent: float = 0,
) -> tuple[int, int, int, int]:
    # Expand the bounding box by padding_percent and padding
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    x1 = max(0, x1 - int(bbox_width * padding_percent) - padding)
    y1 = max(0, y1 - int(bbox_height * padding_percent) - padding)
    x2 = min(w, x2 + int(bbox_width * padding_percent) + padding)
    y2 = min(h, y2 + int(bbox_height * padding_percent) + padding)
    return (x1, y1, x2, y2)


# def img_crop(image: torch.Tensor, width: int, height: int, x: int, y: int):
#     x = min(x, image.shape[2] - 1)
#     y = min(y, image.shape[1] - 1)
#     to_x = min(x + width, image.shape[2])
#     to_y = min(y + height, image.shape[1])
#     return image[:, y:to_y, x:to_x, :]


def get_landmarks(face: Face) -> dict[str, np.ndarray]:
    shape = face[0]["landmark_2d_106"]
    landmarks: np.ndarray = np.round(shape).astype(np.int64)
    return {
        "landmarks": landmarks,
        "main_features": landmarks[33:],
        "left_eye": landmarks[87:97],
        "right_eye": landmarks[33:43],
        "eyes": landmarks[[*range(33, 43), *range(87, 97)]],
        "nose": landmarks[72:87],
        "mouth": landmarks[52:72],
        "left_brow": landmarks[97:106],
        "right_brow": landmarks[43:52],
        "outline": landmarks[[*range(33), *range(48, 51), *range(102, 105)]],
        "outline_forehead": landmarks[[*range(33), *range(48, 51), *range(102, 105)]],
    }


class FACE_CROP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "face_data": ("FACEDATA",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "padding_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("face_crop1", "face_crop2", "face_crop3", "face_crop4")
    FUNCTION = "face_crop"
    CATEGORY = "CJ Face Nodes"

    def face_crop(
        self,
        images: torch.Tensor,  # (B, H, W, C)
        face_data: list[list[FaceData]],
        padding: int,
        padding_percent: float,
    ) -> tuple[torch.Tensor, ...]:
        if len(images) != len(face_data):
            raise ValueError("Number of images must match number of face lists")
        result: list[torch.Tensor] = []
        width = images.shape[2]  # W Dimension
        height = images.shape[1]  # H Dimension
        max_face_count = 4
        face_count = 0

        # Process each image and its faces
        for i, faces in enumerate(face_data):
            if face_count >= max_face_count:
                break

            if not faces:
                continue

            # Process faces in current image
            for face in faces:
                if face_count >= max_face_count:
                    break
                face_count += 1

                x1, y1, x2, y2 = face.bbox
                # Apply padding if needed
                if padding != 0 or padding_percent != 0:
                    x1, y1, x2, y2 = expand_bbox(
                        width, height, face.bbox, padding, padding_percent
                    )

                # Safe cropping with batch dimension
                x = max(0, min(x1, width - 1))
                y = max(0, min(y1, height - 1))
                to_x = min(x + (x2 - x1), width)
                to_y = min(y + (y2 - y1), height)

                if to_x > x and to_y > y:  # Ensure valid crop dimensions
                    cropped_image = images[i : i + 1, y:to_y, x:to_x, :]
                    result.append(cropped_image)

        # If no faces, throw an error
        if not result:
            raise ValueError("No faces found in image")
        return tuple(result)


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


class FACE_MASK_FROM_LANDMARKS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "face_data": ("FACEDATA",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "landmarks": (LANDMARK_REGIONS, {"default": "main_features"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "face_mask"
    CATEGORY = "CJ Face Nodes"

    def face_mask(
        self,
        images: torch.Tensor,  # (B, H, W, C)
        face_data: list[list[FaceData]],
        padding: int,
        feather: int,
        landmarks: str,
    ):
        width = images.shape[2]  # W Dimension
        height = images.shape[1]  # H Dimension
        batch_size = images.shape[0]

        if batch_size != len(face_data):
            raise ValueError("Number of images must match number of face lists")

        # Initialize with correct batch dimension
        masks = torch.zeros((batch_size, height, width))
        masks_cropped = torch.zeros((batch_size, 1024, 1024))

        # Process each image and its faces
        for i, faces in enumerate(face_data):
            if not faces:
                continue

            # Create mask for current image
            mask = np.zeros(
                (height, width), dtype=np.float32
            )  # Use float32 for compatibility with PyTorch
            for face in faces:
                # Get landmarks
                points = cv2.convexHull(face.landmarks[landmarks])
                cv2.fillConvexPoly(mask, points, (1, 1, 1))  # Use color scalar (1,1,1)

            if padding > 0:
                kernel = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

            # Apply feathering
            if feather > 0:
                kernel_size = feather if feather % 2 == 1 else feather + 1
                mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

            # Find the smallest possible square that can contain the mask
            points = cv2.findNonZero(mask)
            if points is not None:
                x, y, w, h = cv2.boundingRect(points)

                # Calculate square parameters
                side_length = max(w, h)
                center_x = x + w // 2
                center_y = y + h // 2

                # Calculate square corners
                half_side = side_length // 2
                x1 = np.clip(center_x - half_side, 0, width)
                y1 = np.clip(center_y - half_side, 0, height)
                x2 = np.clip(center_x + half_side, 0, width)
                y2 = np.clip(center_y + half_side, 0, height)

                # Crop the mask to the square
                cropped_mask = mask[y1:y2, x1:x2]
                masks_cropped[i] = torch.from_numpy(cropped_mask).float()

            # Convert to PyTorch tensor and assign to batch
            masks[i] = torch.from_numpy(mask).float()

        return (masks,)


class FACE_DETAILER_CROP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "face_data": ("FACEDATA",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "landmarks": (LANDMARK_REGIONS, {"default": "main_features"}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "FACECROPDATA")
    RETURN_NAMES = ("face_masks", "face_images", "face_crop_data")
    FUNCTION = "get_face_masks"
    CATEGORY = "CJ Face Nodes"

    def get_face_masks(
        self,
        images: torch.Tensor,  # (B, H, W, C)
        face_data: list[list[FaceData]],  # List of list of FaceData
        padding: int,  # How much to expand mask
        feather: int,  # How much to feather the mask
        landmarks: str,  # Which landmarks to use
    ):
        images_width = images.shape[2]
        images_height = images.shape[1]
        batch_size = images.shape[0]

        # The cropped face image size (Square)
        face_image_size = 1024
        if batch_size != len(face_data):
            raise ValueError("Number of images must match number of face lists")

        # Calculate total number of faces
        total_faces = sum(len(faces) for faces in face_data)
        # The cropped mask will be 1024x1024
        face_masks = torch.zeros((total_faces, face_image_size, face_image_size))
        # Store cropped face images
        face_images = torch.zeros((total_faces, face_image_size, face_image_size, 3))
        # Store face crop data. List of tuples (batch_idx, x1, y1, x2, y2)
        face_crop_data: list[tuple[int, int, int, int, int]] = []

        # Keep track of current face index
        face_index = 0

        # Pre-allocate single numpy array to for faces
        face_mask = np.zeros((face_image_size, face_image_size), dtype=np.float32)

        for batch_idx, faces in enumerate(face_data):
            for face in faces:
                face_mask.fill(0)  # Reset mask
                if (
                    landmarks not in face.landmarks
                    or len(face.landmarks[landmarks]) == 0
                ):
                    continue

                # Get convex hull of landmarks
                hull_points = cv2.convexHull(face.landmarks[landmarks])
                if hull_points is None or len(hull_points) < 3:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(hull_points)

                # Calculate the square size based on the larger dimension
                size = max(w, h)

                # Calculate center point of original bbox
                center_x = x + w // 2
                center_y = y + h // 2

                # Calculate square bounds centered on original bbox
                half_size = size // 2
                x1 = max(0, center_x - half_size - padding)
                y1 = max(0, center_y - half_size - padding)
                x2 = min(images_width, center_x + half_size + padding)
                y2 = min(images_height, center_y + half_size + padding)

                # Ensure final size is square by using smaller dimension
                final_size = min(x2 - x1, y2 - y1)
                x2 = x1 + final_size
                y2 = y1 + final_size

                # Cut image to bounding box
                face_image = images[batch_idx, y1:y2, x1:x2, :]
                # Permute dimensions to (C, H, W) for resize operation
                face_image = face_image.permute(2, 0, 1)
                # Resize image to the face image size
                face_image = T.Resize((face_image_size, face_image_size))(face_image)
                # Permute back to (H, W, C)
                face_image = face_image.permute(1, 2, 0)

                # Double check so the area is a square
                if x2 - x1 != y2 - y1:
                    raise ValueError("Bounding box is not square")

                # Adjust points relative to ROI
                roi_points = hull_points - np.array([[x1, y1]])

                # Calculate how much we need to scale the bounding box to be the face image size
                scale = face_image_size / (x2 - x1)

                # Scale the ROI points
                roi_points = (roi_points * scale).astype(np.int32)

                # Fill the convex polygon onto the mask
                cv2.fillConvexPoly(face_mask, roi_points, (1, 1, 1))

                # Apply feathering if needed
                if feather > 0:
                    kernel_size = feather * 2 + 1
                    face_mask = cv2.GaussianBlur(
                        face_mask, (kernel_size, kernel_size), 0
                    )

                # Convert to tensor and store in batch
                face_masks[face_index] = torch.from_numpy(face_mask)
                face_images[face_index] = face_image
                face_crop_data.append((batch_idx, x1, y1, x2, y2))
                face_index += 1

        return (face_masks, face_images, face_crop_data)


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

        # All face images are 1024x1024 but should be scaled to the bounding
        # box size and placed on the original image according to the crop data

        face_index = 0
        for batch_idx, x1, y1, x2, y2 in face_crop_data:
            face_image = face_images[face_index]
            face_index += 1

            # Ensure the face image is the correct size
            if face_image.shape[0] != 1024 or face_image.shape[1] != 1024:
                raise ValueError("Face image is not the correct size")

            # Scale the face image to the size of the bounding box
            face_image = T.Resize((x2 - x1, y2 - y1))(face_image)
            # Copy the face image onto the original image
            images[batch_idx, y1:y2, x1:x2, :] = face_image
        return (images,)


NODE_CLASS_MAPPINGS = {
    "InsightFaceModelLoader": INSIGHFACE_MODEL_LOADER,
    "GetFaces": GET_FACES,
    "FaceDetailerCrop": FACE_DETAILER_CROP,
    "FaceCrop": FACE_CROP,
    "FaceMaskFromLandMarks": FACE_MASK_FROM_LANDMARKS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InsightFaceModelLoader": "InsightFace Model Loader",
    "GetFaces": "Get Faces",
    "FaceDetailerCrop": "Face Detailer Crop #1",
    "FaceCrop": "Face Crop",
    "FaceMaskFromLandMarks": "Face Mask From LandMarks",
}


# The tensor image will have the format (B, H, W, C), where:
# B is the batch dimension.
# H is the height of the image.
# W is the width of the image.
# C is the number of channels (3 for RGB).
# The values in the tensor will be in the range [0, 1].
