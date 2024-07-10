from .abstract import AbstractMotionAnonymization

import numpy as np


class Trajectory_feature_extractionAnonymization(AbstractMotionAnonymization):
    """Calculates for each point (except the one at the end and start) the difference between the rolling average or
    the interpolated value.

    Required pips:
        none
    Parameters:
        (string) type: either "edges" or "all"
        (int) window_size: 1, window size left and right of the given pose
    """

    name = "trajectory_feature_extraction"

    def validate_config(self):
        if "type" not in self.config:
            self.log.info("Missing type, taking default all")
            self.config["type"] = "all"

        if "window_size" not in self.config:
            self.log.info("Missing window_size value, taking default 1")
            self.config["window_size"] = 1

        if "invert" not in self.config:
            self.log.info("Missing inverted value, taking default False")
            self.config["invert"] = False

    def anonymize(self, mocap, data):
        window_size = self.config["window_size"]
        new_data = []
        for i in range(window_size, len(data) - window_size):
            if self.config["type"] == "all":
                average = np.mean(data[i - window_size : i + window_size], axis=0)

            if self.config["type"] == "edges":
                average = (data[i - window_size] + data[i + window_size]) / 2

            if self.config["invert"]:
                new_data.append(data[i] - average)
            else:
                new_data.append(average)

        padding = np.zeros((len(data) - len(new_data), len(new_data[0]))).tolist()
        new_data = new_data + padding
        new_data = np.array(new_data)

        return new_data
