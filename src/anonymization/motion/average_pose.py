from .abstract import AbstractMotionAnonymization

import numpy as np


class Average_poseAnonymization(AbstractMotionAnonymization):
    """Anonymize by averaging the pose

    Required pips:
        none
    Parameters:
        none
    """

    name = "average_pose"

    def anonymize(self, mocap, data):
        average_pose = np.array([0] * len(data[0]))
        for pose in data:
            average_pose = np.add(average_pose, pose)

        average_pose = np.divide(average_pose, len(data))

        return average_pose
