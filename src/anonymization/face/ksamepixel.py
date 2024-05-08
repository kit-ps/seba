from .kanon import KanonAnonymization

import cv2
import numpy as np


class KsamepixelAnonymization(KanonAnonymization):
    """K-Same-Pixel anonymization

    Required pips:
        - numpy
        - sklearn
        - opencv

    Parameters:
        - (int) k: number of similar identities to choose (-1)
        - (int) pcan: number of components of the PCA
    """

    name = "ksamepixel"

    def merge_images(self, points, shape=()):
        images = list(map(lambda x: cv2.imread(x.get_path()), points))
        return np.mean(images, axis=0)
