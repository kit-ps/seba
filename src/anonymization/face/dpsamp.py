from .abstract import AbstractFaceAnonymization

import cv2
import math
import numpy as np
import random
import scipy.interpolate
from multiprocessing import Pool


def anonymize_image(imgpath, config):
    random.seed(a=config["seed"])
    img = cv2.imread(imgpath)
    # generate k clusters using k-means
    pxls = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, label, center = cv2.kmeans(pxls, config["k"], None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    frequencies = []

    for i in range(config["k"]):
        frequencies.append(0)
        for j in range(pxls.shape[0]):
            if label[j][0] == i:
                if np.absolute(center[i] - pxls[j]).sum() <= config["threshold"]:
                    frequencies[i] += 1

    # calculate budgets and choose sample pixels
    sampled_pixels = []
    for i in range(config["k"]):
        try:
            budget = (config["e"] * frequencies[i]) / sum(frequencies)
            x = 1
            while True:
                x += 1
                v = math.comb(frequencies[i], x) / math.comb(frequencies[i] - config["m"], x)
                if v > math.exp(budget):
                    x -= 1
                    break
            pxls_to_choose = []
            for j in range(pxls.shape[0]):
                if label[j][0] == i:
                    if np.absolute(center[i] - pxls[j]).sum() <= config["threshold"]:
                        pxls_to_choose.append(j)
            random.shuffle(pxls_to_choose)
            sampled_pixels += pxls_to_choose[:x]
        except Exception:
            pass

    # interpolate image based on sample pixels
    xs, ys, colors = [], [], []
    for px in sampled_pixels:
        colors.append(center[label.flatten()[px]])
        xs.append(px // img.shape[0])
        ys.append(px % img.shape[0])
    linimg = [[], [], []]
    nnimg = [[], [], []]
    grid = tuple(np.mgrid[0 : img.shape[0], 0 : img.shape[1]])
    for i in range(3):
        ci = np.array(list(map(lambda x: x[i], colors)))
        linimg[i] = scipy.interpolate.griddata((xs, ys), ci, grid, method="linear", fill_value=0.0)
        (xsn, ysn) = np.nonzero(linimg[i])
        nnimg[i] = scipy.interpolate.griddata((xsn, ysn), linimg[i][xsn, ysn], grid, method="nearest")

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x, y] = [nnimg[0][x][y], nnimg[1][x][y], nnimg[2][x][y]]

    cv2.imwrite(imgpath, img)


class DpsampAnonymization(AbstractFaceAnonymization):
    """Apply a DP-Samp anonymization to the face in an image

    Paper:  Reilly, Dominick, and Liyue Fan. “A Comparative Evaluation of
            Differentially Private Image Obfuscation.” In 2021 Third IEEE
            International Conference on Trust, Privacy and Security in Intelligent
            Systems and Applications (TPS-ISA), 80–89. Atlanta, GA, USA: IEEE, 2021.
            https://doi.org/10.1109/TPSISA52974.2021.00009.


    Required pips:
        - opencv-python

    Parameters:
        - (float) e: privacy budget
        - (int) k: number of clusters
        - (int) m: number of pixels
        - (float) threshold: distance to center for pixel considered to have this intensity
    """

    name = "dpsamp"

    def validate_config(self):
        if "e" not in self.config:
            raise AttributeError("DP-Samp anonymization: missing parameter e (privacy budget)")
        if "k" not in self.config:
            raise AttributeError("DP-Samp anonymization: missing parameter k (number of clusters)")
        if "m" not in self.config:
            raise AttributeError("DP-Samp anonymization: missing parameter m (number of pixels)")
        if "threshold" not in self.config:
            raise AttributeError("DP-Samp anonymization: missing parameter threshold")
        if "seed" not in self.config:
            self.config["seed"] = None

        if "opt" not in self.config or "threads" not in self.config["opt"]:
            self.config["opt"] = {"threads": 24}

    def anonymize_all(self):
        p = Pool(processes=self.config["opt"]["threads"])
        args = list(map(lambda x: (x.get_path(), self.config), self.dataset.datapoints.values()))
        p.starmap(anonymize_image, args)
        p.close()
