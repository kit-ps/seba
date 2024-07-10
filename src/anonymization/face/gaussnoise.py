from .abstract import AbstractFaceAnonymization

import cv2
import numpy as np


class GaussnoiseAnonymization(AbstractFaceAnonymization):
    """Add gaussian noise to the face in an image

    Required pips:
        - opencv-python

    Parameters:
        - (int) sigma: sigma for noise distribution function
    """

    name = "gaussnoise"

    def validate_config(self):
        if "sigma" not in self.config:
            self.config["sigma"] = 10
        else:
            self.config["sigma"] = int(self.config["sigma"])

    def anonymize(self, image):
        img = cv2.imread(image.get_path())
        im = np.zeros(img.shape, np.uint8)
        sigma = (self.config["sigma"], self.config["sigma"], self.config["sigma"])
        cv2.randn(im, (0, 0, 0), sigma)
        img = cv2.add(img, im)

        cv2.imwrite(image.get_path(), img)
