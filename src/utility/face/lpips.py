from ...lib.result import Result
from .abstract import AbstractFaceUtility
from ...lib.inference import Comparison

import lpips
import torch
import cv2
import numpy as np
from torch.nn.functional import interpolate


class LpipsUtility(Comparison, AbstractFaceUtility):
    """This uses Learned Perceptual Image Patch Similarity to compare original and new image

    Paper: Zhang, Richard, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang.
    "The unreasonable effectiveness of deep features as a perceptual metric."
    In Proceedings of the IEEE conference on computer vision and pattern privacy, pp. 586-595. 2018.

    Required pips:
        lpips, torch, opencv, numpy

    Parameters:
        - model (string): model to use for lpips, one of [alex, vgg] (optional, default: alex)
    """

    def validate_config(self):
        if "model" not in self.config:
            self.config["model"] = "alex"

    def init(self):
        self.loss_fn = lpips.LPIPS(net=self.config["model"])

    def load_image(self, image):
        return torch.Tensor((cv2.imread(image.get_path()) / 255.0 / 2.0 - 1.0)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

    def compare_point(self, old_point, new_point):
        img1 = self.load_image(old_point)
        img2 = self.load_image(new_point)

        if img1.shape is not img2.shape:
            img2 = interpolate(img2, size=img1.shape[-1], mode="nearest")

        dist = 1 - float(self.loss_fn(img1, img2))

        rs = Result(old_point.idname, old_point.pointname)
        rs.add_recognized(old_point.idname, dist=dist)
        self.log.debug(old_point.idname + "\t\t" + old_point.pointname + "\t\t" + str(dist))

        return rs
