from .abstract import AbstractFaceAnonymization

import cv2
import numpy as np


class BlackboxAnonymization(AbstractFaceAnonymization):
    """Apply a blackbox anonymization to the face in an image

    Required pips:
        - numpy
        - opencv-python

    Parameters:
        none
    """

    name = "blackbox"

    def anonymize(self, image):
        img = cv2.imread(image.get_path())
        nimg = np.zeros(img.shape)
        cv2.imwrite(image.get_path(), nimg)
