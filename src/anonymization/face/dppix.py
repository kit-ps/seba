from .abstract import AbstractFaceAnonymization

import cv2
import math
import numpy as np


class DppixAnonymization(AbstractFaceAnonymization):
    """Apply DP-Pix to the face in an image

    Paper:  L. Fan. Image pixelization with differential privacy. In F. Kerschbaum
            and S. Paraboschi, editors, Data and Applications Security and Privacy
            XXXII, pages 148–162, Cham, 2018. Springer International Publishing.
            https://par.nsf.gov/servlets/purl/10081774

    Required pips:
        - numpy
        - opencv-python

    Parameters:
        - (float) e: privacy budget
        - (int) b: block size
        - (int) m: number of pixels
    """

    name = "dppix"

    def validate_config(self):
        if "e" not in self.config:
            raise AttributeError("DP-Pix anonymization: missing parameter e (privacy budget)")
        if "b" not in self.config:
            raise AttributeError("DP-Pix anonymization: missing parameter b (block size)")
        if "m" not in self.config:
            raise AttributeError("DP-Pix anonymization: missing parameter m (number of pixels)")

    def anonymize(self, image):
        img = cv2.imread(image.get_path())
        copy = img.copy()
        for i in range(math.ceil(img.shape[0] / self.config["b"])):
            for j in range(math.ceil(img.shape[0] / self.config["b"])):
                colors = []
                for x in range(self.config["b"]):
                    for y in range(self.config["b"]):
                        try:
                            colors.append(img[i * self.config["b"] + x, j * self.config["b"] + y])
                        except IndexError:
                            pass
                new_color = np.mean(colors, axis=0)
                for x in range(3):
                    new_color[x] += np.random.default_rng().laplace(
                        scale=((255 * self.config["m"]) / (self.config["b"] * self.config["b"] * self.config["e"]))
                    )
                    new_color[x] = max(0, min(255, new_color[x]))

                for x in range(self.config["b"]):
                    for y in range(self.config["b"]):
                        try:
                            copy[i * self.config["b"] + x, j * self.config["b"] + y] = new_color
                        except IndexError:
                            pass
        cv2.imwrite(image.get_path(), copy)
