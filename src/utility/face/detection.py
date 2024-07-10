from .abstract import AbstractFaceUtility
from ...lib.inference import Comparison
from ...lib.result import Result

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class DetectionUtility(Comparison, AbstractFaceUtility):
    """This attempts face detection on the anonymized image and returns the confidence score of the detector
            or 0.0 if no face was detected.

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

    def compare_point(self, old_point, new_point):
        image = mp.Image.create_from_file(new_point.get_path())
        detection_result = self.detector.detect(image)

        if len(detection_result.detections):
            score = detection_result.detections[0].categories[0].score
        else:
            score = 0.0

        rs = Result(old_point.idname, old_point.pointname)
        rs.add_recognized(old_point.idname, dist=score)
        return rs
