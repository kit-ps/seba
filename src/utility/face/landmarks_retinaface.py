from .landmarks import LandmarksUtility

import cv2
from retinaface import RetinaFace


class Landmarks_retinafaceUtility(LandmarksUtility):
    """This uses calculates the euclidean distance between face landmarks detected in two images

    Required pips:
        retinaface

    Parameters:
        none
    """

    def init(self):
        self.model = RetinaFace.build_model()

    def get_landmarks(self, image):
        img = cv2.imread(image.get_path())
        faces = RetinaFace.detect_faces(img, threshold=0.2, model=self.model)
        if "face_1" in faces:
            return list(
                map(
                    lambda x: (x[0] / img.shape[0], x[1] / img.shape[1]),
                    map(lambda x: faces["face_1"]["landmarks"][x], ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]),
                )
            )
        else:
            return None
