from .abstract import AbstractMotionAnonymization

import numpy as np


class Motion_extractionAnonymization(AbstractMotionAnonymization):
    """Apply motion extraction anonymization

    Required pips:
        none
    Parameters:
        - (int) difference_size: The index difference of the poses which are subtracted from each other.
    """

    name = "motion_extraction"

    def validate_config(self):
        if "difference" not in self.config:
            self.log.info("Difference not specified, taking default value 1")
            self.config["difference"] = 1

    def anonymize(self, mocap, data):
        new_poses = []
        for i in range(len(data) - self.config["difference"]):
            new_poses.append(data[i + self.config["difference"]] - data[i])

        padding = np.zeros((len(data) - len(new_poses), len(new_poses[0]))).tolist()
        new_poses = new_poses + padding
        new_poses = np.array(new_poses)

        return new_poses
