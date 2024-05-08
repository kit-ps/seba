from .resample import ResampleDeanonymization

import cv2


class LinearDeanonymization(ResampleDeanonymization):
    """De-Anonymize faces by applying linear interpolation

    Required pips:
        - opencv2

    Parameters:
        none
    """

    name = "linear"
    interpolation = cv2.INTER_LINEAR
