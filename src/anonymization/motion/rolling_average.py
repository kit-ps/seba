from .abstract import AbstractMotionAnonymization

import numpy as np


class Rolling_averageAnonymization(AbstractMotionAnonymization):
    """Apply no anonymization

    Required pips:
        none
    Parameters:
        (int) window_size: Number of poses that are in one average window
    """

    name = "rolling_average"

    def validate_config(self):
        if "window_size" not in self.config:
            self.log.info("Missing window_size, taking 2")
            self.config["window_size"] = 2

    def anonymize(self, mocap, data):
        new_poses = []
        for i in range(len(data.T)):
            new_poses.append(np.convolve(data.T[i], np.ones(self.config["window_size"]), "valid") / self.config["window_size"])

        new_poses = np.array(new_poses).T

        return new_poses
