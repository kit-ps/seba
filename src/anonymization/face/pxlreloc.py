from .abstract import AbstractFaceAnonymization

import cv2
import copy


class PxlrelocAnonymization(AbstractFaceAnonymization):
    """Apply a pixel relocation anonymization

    Paper:
        J. Cichowski and A. Czyzewski,
        "Reversible video stream anonymization for video surveillance systems based on pixels relocation and watermarking,"
        2011 IEEE International Conference on Computer Vision Workshops (ICCV Workshops),
        Barcelona, Spain, 2011, pp. 1971-1977, doi: 10.1109/ICCVW.2011.6130490.

    Required pips:
        - opencv-python

    Parameters:
        - steps (int): Number of steps pixel positions are shifted
    """

    name = "pxlreloc"

    def validate_config(self):
        if "steps" not in self.config:
            raise AttributeError("PxlrelocAnonymization: config: Requires number of steps to move pixel position by")

    def anonymize(self, image):
        img = cv2.imread(image.get_path())
        newimg = copy.deepcopy(img)

        forder = self.gen_order(img.shape[0], img.shape[1])
        torder = forder[self.config["steps"] :] + forder[: self.config["steps"]]

        for i in range(len(forder)):
            newimg[torder[i]] = img[forder[i]]

        cv2.imwrite(image.get_path(), newimg)

    def gen_order(self, max, may):
        if max == 0 or may == 0:
            return []
        elif max == 1:
            order = []
            for y in range(may):
                order.append((0, y))
            return order
        elif may == 1:
            order = []
            for x in range(max):
                order.append((x, 0))
            return order
        else:
            order = []
            for x in range(max - 1):
                order.append((x, may - 1))
            for y in range(may):
                order.append((max - 1, may - y - 1))
            for y in range(may - 2):
                order.append((max - 2, y))
            for x in range(max - 1):
                order.append((max - 2 - x, may - 2))
            return order + self.gen_order(max - 2, may - 2)
