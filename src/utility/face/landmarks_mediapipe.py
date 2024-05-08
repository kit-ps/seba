from .landmarks import LandmarksUtility

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class Landmarks_mediapipeUtility(LandmarksUtility):
    """This uses calculates the euclidean distance between face landmarks detected in two images

    Download face landmark detector using:
        wget -O bin/face_detector.tflite -q https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite

    Required pips:
        mediapipe

    Parameters:
        none
    """

    def init(self):
        base_options = python.BaseOptions(model_asset_path="bin/face_detector.tflite")
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.0,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def get_landmarks(self, image):
        image = mp.Image.create_from_file(image.get_path())
        detection_result = self.detector.detect(image)

        if len(detection_result.detections):
            return list(
                map(
                    lambda k: (k.x, k.y),
                    detection_result.detections[0].keypoints,
                )
            )
        else:
            return None
