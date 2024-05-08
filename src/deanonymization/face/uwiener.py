from .abstract import AbstractFaceDeanonymization
from ...privacy.face.ssim import SsimRecognition

import cv2
import numpy as np
from skimage.restoration import unsupervised_wiener


class UwienerDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize using unsupervised Wiener deconvolution

    We use the training data to estimate the kernel size, but not the K value (as in wiener.py)

    Required pips:
        - opencv2
        - skimage
        - numpy

    Parameters:
        none
    """

    name = "uwiener"

    def train(self, clear_set, anon_set):
        if not self.config["ks"] == 0:
            self.log.info("Kernel size Training: Using predefined value " + str(self.config["ks"]))
            return

        if self.name + "-ks" in clear_set.meta:
            self.config["ks"] = int(clear_set.meta[self.name + "-ks"])
            self.log.info("Kernel size Training: Using value from training set metdata: " + str(self.config["ks"]))
            return

        self.log.info("Kernel size Training: Finding optimal value")
        values = list(np.arange(1, 21, 2))
        avgs = []
        rec = SsimRecognition({})

        for i in range(len(values)):
            ks = values[i]
            ssims = []

            for key, clear_point in clear_set.datapoints.items():
                clear_img = cv2.imread(clear_point.get_path())
                anon_img = cv2.imread(anon_set.datapoints[key].get_path())

                psf = np.ones((ks, ks)) / (ks * ks)
                filtered = self.deconv(anon_img, psf)
                ssims.append(rec.ssim(clear_img, filtered))
            mean = np.nanmean(ssims)
            self.log.info("Kernel size Training: value {} -> mean ssim {}".format(ks, mean))
            avgs.append(mean)

        best = np.argmax(avgs)
        best_value = values[best]
        self.config["ks"] = int(best_value)
        self.log.info("Kernel size Training completed. Selected value " + str(self.config["ks"]))

        clear_set.meta[self.name + "-ks"] = self.config["ks"]
        clear_set.save_meta()

    def deanonymize(self, image):
        img = cv2.imread(image.get_path())
        psf = np.ones((self.config["ks"], self.config["ks"])) / (self.config["ks"] * self.config["ks"])
        filtered = self.deconv(img, psf)
        cv2.imwrite(image.get_path(), filtered)

    def deconv(self, img, psf):
        b, g, r = cv2.split(img / 255.0)

        f_b, _ = unsupervised_wiener(b, psf)
        f_g, _ = unsupervised_wiener(g, psf)
        f_r, _ = unsupervised_wiener(r, psf)

        filtered = cv2.merge([f_b, f_g, f_r])
        filtered *= 255.0 / filtered.max()
        return filtered
