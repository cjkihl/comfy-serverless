from insightface.app import FaceAnalysis
import numpy as np
import torchvision.transforms.v2 as T


class INSIGHFACE_TO_ANALYSIS_MODEL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "insightface": ("FACEANALYSIS",),
            },
        }

    FUNCTION = "run"
    CATEGORY = "CJ Nodes"
    RETURN_TYPES = ("ANALYSIS_MODELS",)

    def run(self, insightface: FaceAnalysis):
        return (InsightFaceWrapper(insightface),)


class InsightFaceWrapper:
    def __init__(self, insightface: FaceAnalysis):
        self.face_analysis = insightface
        self.thresholds = {"cosine": 0.68, "euclidean": 4.15, "L2_norm": 1.13}

    def get_face(self, image):
        for size in [(size, size) for size in range(640, 256, -64)]:
            self.face_analysis.det_model.input_size = size
            faces = self.face_analysis.get(image)
            # Sort faces by bounding box size
            if len(faces) > 0:
                return sorted(
                    faces,
                    key=lambda x: (x["bbox"][2] - x["bbox"][0])
                    * (x["bbox"][3] - x["bbox"][1]),
                    reverse=True,
                )
        return None

    def get_embeds(self, image):
        face = self.get_face(image)
        if face is not None:
            face = face[0].normed_embedding
        return face

    def get_bbox(self, image, padding=0, padding_percent=0):
        faces = self.get_face(np.array(image))
        img = []
        x = []
        y = []
        w = []
        h = []
        if faces is not None:
            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                width = x2 - x1
                height = y2 - y1
                x1 = int(max(0, x1 - int(width * padding_percent) - padding))
                y1 = int(max(0, y1 - int(height * padding_percent) - padding))
                x2 = int(min(image.width, x2 + int(width * padding_percent) + padding))
                y2 = int(
                    min(image.height, y2 + int(height * padding_percent) + padding)
                )
                crop = image.crop((x1, y1, x2, y2))
                img.append(T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0))
                x.append(x1)
                y.append(y1)
                w.append(x2 - x1)
                h.append(y2 - y1)

        return (img, x, y, w, h)

    def get_keypoints(self, image):
        face = self.get_face(image)
        if face is not None:
            shape = face[0]["kps"]
            right_eye = shape[0]
            left_eye = shape[1]
            nose = shape[2]
            left_mouth = shape[3]
            right_mouth = shape[4]

            return [left_eye, right_eye, nose, left_mouth, right_mouth]
        return None

    def get_landmarks(self, image, extended_landmarks=False):
        face = self.get_face(image)
        if face is not None:
            shape = face[0]["landmark_2d_106"]
            landmarks = np.round(shape).astype(np.int64)

            main_features = landmarks[33:]
            left_eye = landmarks[87:97]
            right_eye = landmarks[33:43]
            eyes = landmarks[[*range(33, 43), *range(87, 97)]]
            nose = landmarks[72:87]
            mouth = landmarks[52:72]
            left_brow = landmarks[97:106]
            right_brow = landmarks[43:52]
            outline = landmarks[[*range(33), *range(48, 51), *range(102, 105)]]
            outline_forehead = outline

            return [
                landmarks,
                main_features,
                eyes,
                left_eye,
                right_eye,
                nose,
                mouth,
                left_brow,
                right_brow,
                outline,
                outline_forehead,
            ]
        return None


NODE_CLASS_MAPPINGS = {
    "InsightFaceToAnalysisModel": INSIGHFACE_TO_ANALYSIS_MODEL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InsightFaceToAnalysisModel": "InsightFace to Analysis Model",
}
