from .abstract import AbstractFaceDeanonymization

import cv2
import numpy as np


class Interpolate_emptyDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize faces with blacked out pixels by interpolating these from their neighbors

    Required pips:
        - face_recognition (if image metadata does not include bbox)
        - opencv-python
        - numpy

    Parameters:
        - (int) min_confidence: minimum confidence required to fill in pixel [1, 8] (optional, default: 4)
        - (float) threshold: avg rgb value to consider a pixel blacked-out (optional, default: 0.0)
        - (int) empty: which color of pixel to consider empty (optional, default 0 (=black))
    """

    name = "interpolate_empty"

    def validate_config(self):
        if "min_confidence" not in self.config:
            self.config["min_confidence"] = 4
        else:
            self.config["min_confidence"] = int(self.config["min_confidence"])

        if "threshold" not in self.config:
            self.config["threshold"] = 0.0
        else:
            self.config["threshold"] = float(self.config["threshold"])

        if "empty" not in self.config:
            self.config["empty"] = 0
        else:
            self.config["empty"] = int(self.config["empty"])
        self.black = np.array((self.config["empty"], self.config["empty"], self.config["empty"]))

    def deanonymize(self, image):
        img = cv2.imread(image.get_path())
        fixed = 1
        while fixed > 0:
            fixed = 0
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    if self.is_black(img[x, y]):
                        neighbors = []
                        for of_x, of_y in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]:
                            try:
                                ngh = img[x + of_x, y + of_y]
                            except IndexError:
                                ngh = self.black
                            if not self.is_black(ngh):
                                neighbors.append(ngh)

                        if len(neighbors) >= self.config["min_confidence"]:
                            for i in range(3):
                                img[x, y][i] = int(sum(map(lambda x: x[i], neighbors)) / len(neighbors))
                            if not self.is_black(img[x, y]):
                                fixed += 1
        cv2.imwrite(image.get_path(), img)

    def is_black(self, pixel):
        return (sum(abs(pixel - self.black)) / 3) <= self.config["threshold"]
