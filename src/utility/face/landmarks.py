from ...lib.result import Result
from .abstract import AbstractFaceUtility
from ...lib.inference import Comparison

import numpy as np


class LandmarksUtility(Comparison, AbstractFaceUtility):
    """This uses calculates the euclidean distance between face landmarks detected in two images

    Abstract!

    Required pips:
        none

    Parameters:
        none
    """

    def compare_point(self, old_point, new_point):
        l1 = self.get_landmarks(old_point)
        l2 = self.get_landmarks(new_point)

        if l1 and l2:
            dists = []
            for a, b in zip(l1, l2):
                dists.append(np.linalg.norm(np.array(a) - np.array(b)))
            dist = float(np.mean(dists))
        elif l1 and (not l2):
            dist = 1.0
        else:
            dist = np.nan

        rs = Result(old_point.idname, old_point.pointname)
        rs.add_recognized(old_point.idname, dist=dist)
        self.log.debug(old_point.idname + "\t\t" + old_point.pointname + "\t\t" + str(dist))

        return rs
