from ...lib.result import Result
from .abstract import AbstractMotionUtility
from ...lib.inference import Comparison

import numpy as np
from dtw import dtw
from frechetdist import frdist


class Time_series_distanceUtility(Comparison, AbstractMotionUtility):
    """This uses different time-series distances to compare original and new motion

    Required pips:
        frechetdist, dtw

    Parameters:
        - metric (string): metric to compare the two time series, "dtw", "frechet", or "euclidian".
        - step (int): optional parameter for "frechet" distance which reduces the number of curve elements
        by only taking every nth element
    """

    def validate_config(self):
        if "metric" not in self.config:
            self.config["metric"] = "euclidian"
            self.log.info("Missing parameter value for metric, selecting default value 'euclidian'")

        if self.config["metric"] == "frechet" and "step" not in self.config:
            self.config["step"] = 1
            self.log.info("Missing parameter value for step, selecting default value '1'")

    def compare_point(self, old_point, new_point):
        old_mocap = old_point.load().T
        new_mocap = new_point.load().T

        assert old_mocap.shape == new_mocap.shape, "The two time series have different shape and cannot be compared"

        average_distance = 0
        for i in range(len(old_mocap)):
            old_trajectory = old_mocap[i]
            new_trajectory = new_mocap[i]

            if self.config["metric"] == "dtw":
                average_distance += dtw(old_trajectory, new_trajectory, keep_internals=True).normalizedDistance

            if self.config["metric"] == "frechet":
                reduced_old = old_trajectory[0 : -1 : self.config["step"]]
                reduced_new = new_trajectory[0 : -1 : self.config["step"]]
                tmp_old = [[reduced_old[i], i] for i in range(len(reduced_old))]
                tmp_new = [[reduced_new[i], i] for i in range(len(reduced_new))]

                average_distance += frdist(tmp_old, tmp_new)

            if self.config["metric"] == "euclidian":
                average_distance += np.linalg.norm(old_trajectory - new_trajectory)

        average_distance /= len(old_mocap)

        rs = Result(old_point.idname, old_point.pointname)
        rs.add_recognized(old_point.idname, dist=average_distance)
        self.log.debug(old_point.idname + "\t\t" + old_point.pointname + "\t\t" + str(average_distance))

        return rs
