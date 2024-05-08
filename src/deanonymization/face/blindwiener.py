from .abstract import AbstractFaceDeanonymization

import cv2
from skimage import restoration
from scipy.signal import convolve2d
import numpy as np


class BlindwienerDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize faces by applying a Wiener filter using the skimage implementation

    This implementation does not use the training data and is therefore considered blind.

    https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.wiener

    Required pips:
        - opencv2
        - skimage
        - numpy

    Parameters:
        none
    """

    name = "blindwiener"

    def deanonymize(self, image):
        img = cv2.imread(image.get_path())
        b, g, r = cv2.split(img)

        f_b = self.wiener(b)
        f_g = self.wiener(g)
        f_r = self.wiener(r)

        filtered = cv2.merge([f_b, f_g, f_r])

        cv2.imwrite(image.get_path(), filtered)

    def wiener(self, img):
        psf = np.ones((5, 5)) / 25
        img = convolve2d(img, psf, "same")
        rng = np.random.default_rng()
        img += 0.1 * img.std() * rng.standard_normal(img.shape)
        deconvolved_img = restoration.wiener(img, psf, 1100, clip=False)
        return deconvolved_img
