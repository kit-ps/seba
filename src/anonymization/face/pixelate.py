from .abstract import AbstractFaceAnonymization

import cv2


class PixelateAnonymization(AbstractFaceAnonymization):
    """Apply a pixelation anonymization to the face in an image

    Required pips:
        - opencv-python

    Parameters:
        - (int) size: number of pixels on horizontal axis
        - (bool) keepshape: whether to re-upscale the image to its original shape (squares of pixels will have the same color)
                                - useful when images should always have specific size (optional, default: True)
    """

    name = "pixelate"

    def validate_config(self):
        if "size" not in self.config:
            self.config["size"] = 5
        else:
            self.config["size"] = int(self.config["size"])

        if "keepshape" not in self.config:
            self.config["keepshape"] = True
        else:
            self.config["keepshape"] = bool(self.config["keepshape"])

    def anonymize(self, image):
        img = cv2.imread(image.get_path())
        orig_shape = img.shape[:2]
        img = cv2.resize(img, (self.config["size"], int((img.shape[1] / img.shape[0]) * self.config["size"])))
        if self.config["keepshape"]:
            img = cv2.resize(img, orig_shape[:2], interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(image.get_path(), img)
