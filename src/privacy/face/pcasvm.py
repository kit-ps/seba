from ...lib.result import Result
from .abstract import AbstractFacePrivacy
from ...lib.inference import Classification

import cv2
import numpy as np

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class PcasvmClassification(Classification, AbstractFacePrivacy):
    """Train and use a PCA + SVM.

    Based on https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_eigenfaces.html

    Required pips:
        sklearn

    Parameters:
        - (int) pca__n_components
        - (bool) pca__whiten
        - (float) svm__C
        - (none/string) svm__class_weight: one of [None, "balanced"]
        - (string) svm__kernel": one of ["linear", "poly", "rbf"]
    """

    def enroll(self, set):
        self.clf = Pipeline([("scaler", StandardScaler()), ("pca", PCA()), ("svm", SVC())])
        self.clf.set_params(**self.config)

        img, pred = self.load_data(set)
        self.clf.fit(img, pred)

    def classify_all(self, set, results):
        imgs, _ = self.load_data(set)
        preds = self.clf.decision_function(imgs)

        for image, i in zip(set.datapoints.values(), range(len(imgs))):
            rs = Result(image.idname, image.pointname)
            for cls, prob in zip(self.clf.classes_, preds[i]):
                rs.add_recognized(cls, dist=(1 - prob))
            self.log.debug(str(rs))
            results.append(rs)
        return results

    def load_data(self, set):
        img = []
        pred = []
        for point in set.datapoints.values():
            img.append(self.load_img(point))
            pred.append(point.idname)
        return np.array(img), np.array(pred)

    def load_img(self, img):
        img = cv2.imread(img.get_path())
        return img.flatten()
