from .abstract import AbstractFaceAnonymization

import cv2
import random


class DpsnowAnonymization(AbstractFaceAnonymization):
    """Apply DP-Snow to the face in an image

    Paper:  B. John, A. Liu, L. Xia, S. Koppal, and E. Jain. Let it snow: Adding
            pixel noise to protect the user’s identity. In ACM Symposium on Eye
            Tracking Research and Applications, ETRA ’20 Adjunct, New York,
            NY, USA, 2020. Association for Computing Machinery.

    Required pips:
        - opencv-python

    Parameters:
        - (float) d: privacy budget
    """

    name = "dpsnow"

    def init(self):
        random.seed(a=self.config["seed"])

    def validate_config(self):
        if "d" not in self.config:
            raise AttributeError("DP-Snow anonymization: missing parameter d (privacy budget)")
        if "seed" not in self.config:
            self.config["seed"] = None

    def anonymize(self, image):
        img = cv2.imread(image.get_path())
        coords = [(y, x) for x in range(img.shape[1]) for y in range(img.shape[0])]
        random.shuffle(coords)
        for coord in coords[: int((1 - self.config["d"]) * len(coords))]:
            img[coord[0], coord[1]] = [127, 127, 127]
        cv2.imwrite(image.get_path(), img)
