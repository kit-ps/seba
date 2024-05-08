from .abstract import AbstractFaceAnonymization

import cv2
import math


class GaussianblurAnonymization(AbstractFaceAnonymization):
    """Apply a gaussian blur anonymization to the face in an image

    Required pips:
        - opencv-python

    Parameters:
        - (int) kernel: size of the gaussian kernel for faces with a width of 100px
    """

    name = "gaussianblur"

    def validate_config(self):
        if "kernel" not in self.config:
            self.config["kernel"] = 9
        else:
            self.config["kernel"] = int(self.config["kernel"])

    def anonymize(self, image):
        img = cv2.imread(image.get_path())
        # config['kernel'] is for face width 100. linear in face width. round to nearest odd int
        kernel = 2 * math.floor(((img.shape[1]) * self.config["kernel"]) / 200) + 1
        img = cv2.GaussianBlur(img, (kernel, kernel), 0)
        cv2.imwrite(image.get_path(), img)
