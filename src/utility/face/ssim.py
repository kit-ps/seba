from ...lib.result import Result
from .abstract import AbstractFaceUtility
from ...lib.inference import Comparison

import cv2
import numpy as np


class SsimUtility(Comparison, AbstractFaceUtility):
    """This is an implementation of the algorithm for calculating the
    Structural SIMilarity (SSIM) index between two images.

    Paper: Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli,
    "Image quality assessment: From error measurement to structural similarity"
    IEEE Transactios on Image Processing, vol. 13, no. 1, Jan. 2004.

    Required pips:
        numpy, opencv

    Parameters:
        None

    """

    def compare_point(self, old_point, new_point):
        img1 = cv2.imread(old_point.get_path())
        img2 = cv2.imread(new_point.get_path())

        if img1.shape[0] != img2.shape[0] or img1.shape[1] != img2.shape[1]:
            img1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_NEAREST)
            img2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_NEAREST)

        gray1 = self.reduce_dim(img1)
        gray2 = self.reduce_dim(img2)

        dist = self.ssim(gray1, gray2)

        rs = Result(old_point.idname, old_point.pointname)
        rs.add_recognized(old_point.idname, dist=dist)
        self.log.debug(old_point.idname + "\t\t" + old_point.pointname + "\t\t" + str(dist))

        return rs

    def reduce_dim(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def ssim(self, img1, img2, K=None, window=None, L=None):
        if K is None:
            K = (0.01, 0.03)
        if window is None:
            window = (11, 1.5)
        if L is None:
            L = 255

        C1 = (K[0] * L) ** 2
        C2 = (K[1] * L) ** 2

        img1 = img1.astype(float)
        img2 = img2.astype(float)

        mu1 = self.cutoff_gaussblur(img1, window[0], window[1])
        mu2 = self.cutoff_gaussblur(img2, window[0], window[1])

        mu1_sq = np.multiply(mu1, mu1)
        mu2_sq = np.multiply(mu2, mu2)
        mu1_mu2 = np.multiply(mu1, mu2)

        sigma1_sq = self.cutoff_gaussblur(np.multiply(img1, img1), window[0], window[1]) - mu1_sq
        sigma2_sq = self.cutoff_gaussblur(np.multiply(img2, img2), window[0], window[1]) - mu2_sq
        sigma12 = self.cutoff_gaussblur(np.multiply(img1, img2), window[0], window[1]) - mu1_mu2

        ssim_map = np.divide(
            np.multiply((2 * mu1_mu2 + C1), (2 * sigma12 + C2)), np.multiply((mu1_sq + mu2_sq + C1), (sigma1_sq + sigma2_sq + C2))
        )

        return np.mean(ssim_map)

    def cutoff_gaussblur(self, img, kernel, sigma):
        cutoff = int((kernel - 1) / 2)
        m = cv2.GaussianBlur(img, (kernel, kernel), sigma)
        return [x[cutoff:-cutoff] for x in m[cutoff:-cutoff]]
