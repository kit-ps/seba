from .abstract import AbstractFaceDeanonymization
from ...privacy.face.ssim import SsimRecognition

import cv2
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian
import numpy as np


class WienerDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize faces by applying a Wiener filter

    Here, parameters for the filter are estimated based on training data.
    The parameters which achieve the best SSIM between de-anonymized and clear image in training are chosen for test.

    The implementation for the filter itself is based on:
    https://github.com/tranleanh/wiener-median-comparison/blob/master/Wiener_Filter.py

    Required pips:
        - opencv2
        - skimage
        - numpy

    Parameters:
        none
    """

    name = "wiener"

    def train(self, clear_set, anon_set):
        self.log.info("Wiener Filter Training: Finding optimal value for kernel size")
        K = 10
        values = list(np.arange(3, 15, 2))
        avgs = []
        rec = SsimRecognition({})

        for i in range(len(values)):
            kernel_size = values[i]
            ssims = []

            for key, clear_point in list(clear_set.datapoints.items())[:500]:
                clear_img = cv2.imread(clear_point.get_path())
                anon_img = cv2.imread(anon_set.datapoints[key].get_path())
                filtered = self.filter(anon_img, kernel_size, K)
                ssims.append(rec.ssim(clear_img, filtered))

            mean = np.nanmean(ssims)
            self.log.info("Wiener Filter Training: value {} -> mean ssim {}".format(kernel_size, mean))
            avgs.append(mean)

        self.kernel_size = values[np.argmax(avgs)]
        self.log.info("Selected value for kernel size: " + str(self.kernel_size))

        self.log.info("Wiener Filter Training: Finding optimal value for K")
        values = list(np.arange(1, 40, 2))
        avgs = []

        for i in range(len(values)):
            K = values[i]
            ssims = []

            for key, clear_point in list(clear_set.datapoints.items())[:500]:
                clear_img = cv2.imread(clear_point.get_path())
                anon_img = cv2.imread(anon_set.datapoints[key].get_path())
                filtered = self.filter(anon_img, self.kernel_size, K)
                ssims.append(rec.ssim(clear_img, filtered))

            mean = np.nanmean(ssims)
            self.log.info("Resample Training: value {} -> mean ssim {}".format(K, mean))
            avgs.append(mean)

        self.K = values[np.argmax(avgs)]
        self.log.info("Wiener Filter Training completed. Selected value for K: " + str(self.K))

    def deanonymize(self, image):
        img = cv2.imread(image.get_path())
        filtered = self.filter(img, self.kernel_size, self.K)
        cv2.imwrite(image.get_path(), filtered)

    def filter(self, img, kernel_size, K):
        b, g, r = cv2.split(img)

        kernel = self.gaussian_kernel(kernel_size)
        f_b = self.wiener(b, kernel, K)
        f_g = self.wiener(g, kernel, K)
        f_r = self.wiener(r, kernel, K)

        filtered = cv2.merge([f_b, f_g, f_r])
        filtered *= 255.0 / filtered.max()
        return filtered

    def gaussian_kernel(self, kernel_size=3):
        h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
        h = np.dot(h, h.transpose())
        h /= np.sum(h)
        return h

    def wiener(self, img, kernel, K):
        kernel /= np.sum(kernel)
        dummy = np.copy(img)
        dummy = fft2(dummy)
        kernel = fft2(kernel, s=img.shape)
        kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
        dummy = dummy * kernel
        dummy = np.abs(ifft2(dummy))
        return dummy
