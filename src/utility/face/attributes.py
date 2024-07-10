from ...lib.result import Result
from .abstract import AbstractFaceUtility
from ...lib.inference import Comparison
from ...lib.utils import suppress_stdout

import numpy as np
from deepface import DeepFace


class AttributesUtility(Comparison, AbstractFaceUtility):
    """This uses calculates the distance between the detected face attributes of two images

    Required pips:
        deepface

    Parameters:
        none
    """

    def get_attributes(self, point):
        with suppress_stdout():
            return DeepFace.analyze(
                img_path=point.get_path(),
                detector_backend="retinaface",
                already_normalized=True,
                prog_bar=False,
            )

    def rmse(self, d1, d2):
        a = np.array(list(d1.values()))
        b = np.array(list(d2.values()))
        return np.sqrt(np.mean((a - b) ** 2))

    def compare_point(self, old_point, new_point):
        l1 = self.get_attributes(old_point)
        l2 = self.get_attributes(new_point)

        dists = {}
        dists["emotion"] = self.rmse(l1["emotion"], l2["emotion"])
        dists["race"] = self.rmse(l1["race"], l2["race"])
        dists["sex"] = 0 if l1["gender"] == l2["gender"] else 100
        dists["age"] = abs(l1["age"] - l2["age"])

        dist = np.mean(list(dists.values())) / 100

        rs = Result(old_point.idname, old_point.pointname)
        rs.add_recognized(old_point.idname, dist=dist)
        self.log.debug(old_point.idname + "\t\t" + old_point.pointname + "\t\t" + str(dist))

        return rs
