from ...lib.result import Result
from .abstract import AbstractFacePrivacy
from ...lib.inference import Classification

import face_recognition
import numpy as np
import math
import cv2
from sklearn import neighbors


class FrknnClassification(Classification, AbstractFacePrivacy):
    """Use the face_recognition (KNN) privacy method (based on dlib).
    face_recognition documentation: https://github.com/ageitgey/face_recognition

    Required pips:
        - face_recognition
        - sklearn

    NOTE: This implementation is based on this example:
        https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py

    Parameters:
        None
    """

    def validate_config(self):
        pass

    def enroll(self, set):
        self.folder = set.folder
        self.log.info("Starting privacy.\n\tFolder: " + self.folder + "\n\tConfiguration: " + str(self.config))

        self.log.debug("Extracting face encodings.")
        encodings = []
        ids = []
        for point in set.datapoints.values():
            face = face_recognition.load_image_file(point.get_path())
            face_encodings = face_recognition.face_encodings(face)
            if len(face_encodings) == 1:
                encodings.append(face_encodings[0])
                ids.append(point.idname)

        self.log.debug("Training model.")
        n_neighbors = int(round(math.sqrt(len(encodings))))
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="ball_tree", weights="distance")
        if len(encodings):
            self.no_encodings = False
            self.clf.fit(encodings, ids)
        else:
            self.no_encodings = True

    def classify_point(self, image):
        rs = Result(image.idname, image.pointname)

        img = cv2.imread(image.get_path())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)

        for i in range(3):
            if len(encodings) == 0:
                img = cv2.resize(img, list(map(lambda x: 2 * x, img.shape[:2])), interpolation=cv2.INTER_NEAREST)
                encodings = face_recognition.face_encodings(img)

        if len(encodings) == 1 and not self.no_encodings:
            results = np.average(self.clf.predict_proba(encodings), axis=0)
            for i in range(len(results)):
                rs.add_recognized(self.clf.classes_[i], dist=(1 - results[i]))

        self.log.debug(str(rs))
        return rs

    def get_encoding(self, point):
        face = face_recognition.load_image_file(point.get_path())
        return face_recognition.face_encodings(face)
