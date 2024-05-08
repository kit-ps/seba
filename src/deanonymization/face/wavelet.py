from .abstract import AbstractFaceDeanonymization

import cv2
from skimage.restoration import denoise_wavelet


class WaveletDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize using Wavelet denoising

    Required pips:
        - opencv2
        - skimage
        - numpy

    Parameters:
        none
    """

    name = "wavelet"

    def deanonymize(self, image):
        img = cv2.imread(image.get_path())
        filtered = self.deconv(img)
        cv2.imwrite(image.get_path(), filtered)

    def deconv(self, img):
        b, g, r = cv2.split(img / 255.0)

        f_b = denoise_wavelet(b)
        f_g = denoise_wavelet(g)
        f_r = denoise_wavelet(r)

        filtered = cv2.merge([f_b, f_g, f_r])
        filtered *= 255.0 / filtered.max()
        return filtered
