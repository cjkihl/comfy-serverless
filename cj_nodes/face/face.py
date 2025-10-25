from typing import Any
import numpy as np
from numpy.typing import NDArray
import torch
import cv2
import comfy.utils


class FaceData:
    def __init__(self, bbox: np.ndarray, landmarks: np.ndarray, kps: np.ndarray):
        x1, y1, x2, y2 = bbox
        self.bbox: tuple[int, int, int, int] = (int(x1), int(y1), int(x2), int(y2))

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
        self.kps: dict[str, tuple[int, int]] = {
            "left_eye": kps[1],
            "right_eye": kps[0],
            "nose": kps[2],
            "left_mouth": kps[3],
            "right_mouth": kps[4],
        }


def face_data_to_dict(d: FaceData) -> dict[str, list[int]]:
    """Convert FaceData to a dictionary"""
    result = {"bbox": [int(x) for x in d.bbox]}

    # Convert landmarks separately
    for key, value in d.landmarks.items():
        result[key] = []
        for point in value:
            result[key].append(point.tolist())
    return result


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


def calculate_square_bounds(
    w: int,
    h: int,
    center_x: int,
    center_y: int,
    padding_percent: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Calculate square bounds with padding around center point"""
    size = max(w, h)
    center = np.array([center_x, center_y])

    half_size = size // 2
    padding = half_size * (padding_percent / 100)
    padded_half_size = int(half_size + padding)

    x1, y1 = np.maximum(0, center - padded_half_size)
    x2, y2 = np.minimum(
        np.array([image_width, image_height]), center + padded_half_size
    )

    # Ensure square
    final_size = min(x2 - x1, y2 - y1)
    x2, y2 = x1 + final_size, y1 + final_size

    return int(x1), int(y1), int(x2), int(y2)


def dilate_mask(face_mask: np.ndarray, expand_percent: float) -> np.ndarray:
    """
    Dilate mask by percentage of mask size
    Args:
        face_mask: Input mask array (2D binary mask)
        expand_percent: Percentage to expand (0-100)
    Returns:
        Dilated mask array
    """
    if expand_percent <= 0:
        return face_mask

    mask_height, mask_width = face_mask.shape
    kernel_size = int((min(mask_height, mask_width) * expand_percent) / 100)
    # Ensure odd kernel size >= 3
    kernel_size = max(3, kernel_size + (kernel_size % 2 == 0))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    return cv2.dilate(face_mask, kernel)


def feather_mask(face_mask: np.ndarray, feather_percent: float) -> np.ndarray:
    """
    Feather mask edges by percentage of mask size
    Args:
        face_mask: Input mask array (2D binary mask)
        feather_percent: Percentage to feather (0-100)
    Returns:
        Feathered mask array
    """
    if not isinstance(face_mask, np.ndarray) or face_mask.ndim != 2:
        raise ValueError("face_mask must be 2D numpy array")
    if not 0 <= feather_percent <= 100:
        raise ValueError("feather_percent must be between 0 and 100")

    if feather_percent <= 0:
        return face_mask

    mask_height, mask_width = face_mask.shape
    kernel_size = int((min(mask_height, mask_width) * feather_percent) / 100)
    # Ensure odd kernel size >= 3
    kernel_size = max(3, kernel_size + (1 - kernel_size % 2))

    return cv2.GaussianBlur(face_mask, (kernel_size, kernel_size), 0)


def resize_face_image(
    face_image: torch.Tensor, target_size: int, method: str = "bilinear"
) -> torch.Tensor:
    """
    Resize face image tensor to target size
    Args:
        face_image: Input tensor [H,W,C]
        target_size: Output size (square)
        method: Interpolation method
    Returns:
        Resized tensor [target_size,target_size,C]
    """
    # Convert to channels-first format
    face_image = face_image.permute(2, 0, 1)  # [C, H, W]
    face_image = face_image.unsqueeze(0)  # [1, C, H, W]

    # Upscale
    face_image = comfy.utils.common_upscale(
        face_image, target_size, target_size, method, "disabled"
    )

    # Convert back to channels-last format
    face_image = face_image.squeeze(0)  # [C, size, size]
    face_image = face_image.permute(1, 2, 0)  # [size, size, C]

    return face_image

