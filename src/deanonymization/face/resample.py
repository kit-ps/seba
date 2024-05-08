from .abstract import AbstractFaceDeanonymization
from ...privacy.face.ssim import SsimRecognition

import cv2
import numpy as np


class ResampleDeanonymization(AbstractFaceDeanonymization):
    """De-Anonymize faces by applying an interpolation

    If the final resolution matches the input resolution, images are first downsampled to an intermediate resolution.
    Then, this intermediate image is upsampled to the final resolution using a pre-defined interpolation method.
    The intermediate resolution may either be pre-defined, or be learned from training data.
    Learning means testing a variety of resolutions and choosing the one that results in the highest SSIM of de-anonymized and clear images.
    The learned intermediate resolution is also saved to the training data sets and will be used instead of re-training.

    This is an abstract class that is implemented by linear.py and bicubic.py

    Required pips:
        - opencv2
        - numpy

    Parameters:
        none
    """

    name = "resample"
    interpolation = False

    def validate_config(self):
        if "resample" not in self.config:
            self.config["resample"] = 0.0
        else:
            self.config["resample"] = float(self.config["resample"])

        if "final_res" not in self.config:
            self.config["final_res"] = False
        else:
            self.config["final_res"] = int(self.config["final_res"])

    def train(self, clear_set, anon_set):
        if not self.config["resample"] == 0.0:
            self.log.info("Resample Training: Using predefined value " + str(self.config["resample"]))
            return

        if self.name + "-resample" in clear_set.meta:
            self.config["resample"] = float(clear_set.meta[self.name + "-resample"])
            self.log.info("Resample Training: Using value from training set metdata: " + str(self.config["resample"]))
            return

        self.log.info("Resample Training: Finding optimal value")
        values = list(np.arange(0.1, 0.9, 0.05))
        avgs = []
        rec = SsimRecognition({})

        for i in range(len(values)):
            res = values[i]
            ssims = []

            for key, clear_point in clear_set.datapoints.items():
                clear_img = cv2.imread(clear_point.get_path())
                anon_img = cv2.imread(anon_set.datapoints[key].get_path())

                w, h = anon_img.shape[:2]
                r1 = cv2.resize(anon_img, (int(res * w), int(res * h)), interpolation=cv2.INTER_AREA)
                if self.config["final_res"]:
                    w = self.config["final_res"]
                    h = self.config["final_res"]
                r2 = cv2.resize(r1, (w, h), interpolation=self.interpolation)

                g1 = rec.reduce_dim(clear_img)
                g2 = rec.reduce_dim(r2)

                ssims.append(rec.ssim(g1, g2))
            mean = np.nanmean(ssims)
            self.log.info("Resample Training: value {} -> mean ssim {}".format(res, mean))
            avgs.append(mean)

        best = np.argmax(avgs)
        best_value = values[best]
        self.config["resample"] = float(best_value)
        self.log.info("Resample Training completed. Selected value " + str(self.config["resample"]))

        clear_set.meta[self.name + "-resample"] = self.config["resample"]
        clear_set.save_meta()

    def deanonymize(self, image):
        img = cv2.imread(image.get_path())
        w, h = img.shape[:2]

        res = self.config["resample"]
        r1 = cv2.resize(img, (int(res * w), int(res * h)), interpolation=cv2.INTER_AREA)

        if self.config["final_res"]:
            w = self.config["final_res"]
            h = self.config["final_res"]
        r2 = cv2.resize(r1, (w, h), interpolation=self.interpolation)

        cv2.imwrite(image.get_path(), r2)
