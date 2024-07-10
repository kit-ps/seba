from ...lib.result import Result
from .abstract import AbstractFacePrivacy
from ...lib.inference import Classification
from src.utility.face.ssim import SsimUtility

import cv2


class SsimClassification(Classification, AbstractFacePrivacy):
    """This is an image classication using SSIM.

    We simly choose the identity of the image with the highest SSIM from the enrollment set as the identity to classify.

    Paper: Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli,
    "Image quality assessment: From error measurement to structural similarity"
    IEEE Transactios on Image Processing, vol. 13, no. 1, Jan. 2004.

    Parameters:
        none
    """

    def enroll(self, set):
        self.set = set
        self.log.info("Starting privacy.\n\tEnroll-Set: " + self.set.name)
        self.ssim = SsimUtility({})

    def classify_point(self, image):
        rs = Result(image.idname, image.pointname)
        img1 = cv2.imread(image.get_path())
        for galimg in self.set.datapoints.values():
            img2 = cv2.imread(galimg.get_path())
            dist = self.ssim.ssim(img1, img2)
            rs.add_recognized(galimg.idname + "." + galimg.pointname, dist=(1 - dist))
        self.log.debug(str(rs))
        return rs
