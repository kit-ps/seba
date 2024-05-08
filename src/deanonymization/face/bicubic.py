from .resample import ResampleDeanonymization

import cv2


class BicubicDeanonymization(ResampleDeanonymization):
    """De-Anonymize faces by applying bicubic interpolation

    Required pips:
        - opencv2

    Parameters:
        none
    """

    name = "bicubic"
    interpolation = cv2.INTER_CUBIC
