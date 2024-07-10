from .pcasvm import PcasvmClassification

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LandmarksClassification(PcasvmClassification):
    """Train and use a privacy based on detecting face landmarks and using Scalar+PCA+SVM on it.

    Based on https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_eigenfaces.html

    Download face landmark detector using:
        wget -O bin/face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

    Required pips:
        sklearn
        mediapipe

    Parameters:
        - (int) pcan: number of components of the PCA
    """

    def __init__(self, config):
        super().__init__(config)
        base_options = python.BaseOptions(model_asset_path="bin/face_landmarker_v2_with_blendshapes.task")
        options = vision.FaceLandmarkerOptions(
            base_options=base_options, num_faces=1, min_face_detection_confidence=0.0, min_face_presence_confidence=0.0
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def load_img(self, img):
        image = mp.Image.create_from_file(img.get_path())
        detection_result = self.detector.detect(image)

        landmarks = []
        for lm in detection_result.face_landmarks[0]:
            landmarks.append([lm.x, lm.y, lm.z])

        return (landmarks - np.mean(landmarks, axis=0)).flatten()
