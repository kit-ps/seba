from .kanon import KanonAnonymization

import cv2
import numpy as np


class KsameeigenAnonymization(KanonAnonymization):
    """K-Same-Eigen anonymization

    Required pips:
        - numpy
        - sklearn
        - opencv

    Parameters:
        - (int) k: number of similar identities to choose (-1)
        - (int) pcan: number of components of the PCA
    """

    name = "ksameeigen"

    def merge_images(self, points, shape=()):
        images = list(map(lambda x: cv2.imread(x.get_path()).flatten(), points))
        features = self.pca.transform(self.scaler.transform(images))
        mean = np.mean(features, axis=0)
        img = self.scaler.inverse_transform(self.pca.inverse_transform(mean.reshape(1, -1)))
        return img.reshape(shape)
