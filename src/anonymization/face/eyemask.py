from .abstract import AbstractFaceAnonymization

import cv2


class EyemaskAnonymization(AbstractFaceAnonymization):
    """Apply a mask on the eye area of the face in an image

    Required pips:
        - opencv-python

    Parameters:
        - center (int): y-coordinate of the center of the blackbox
        - dist (int): y-length from center of the blackbox
    """

    name = "eyemask"

    def validate_config(self):
        if "center" not in self.config:
            self.config["center"] = 85
        if "dist" not in self.config:
            self.config["dist"] = 70

    def anonymize(self, image):
        img = cv2.imread(image.get_path())
        img = cv2.rectangle(
            img,
            (0, self.config["center"] - self.config["dist"]),
            (img.shape[1], self.config["center"] + self.config["dist"]),
            (0, 0, 0),
            -1,
        )
        cv2.imwrite(image.get_path(), img)
