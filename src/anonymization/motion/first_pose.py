from .abstract import AbstractMotionAnonymization


class First_poseAnonymization(AbstractMotionAnonymization):
    """Return only the first pose

    Required pips:
        none
    Parameters:
        none
    """

    name = "first_pose"

    def anonymize(self, mocap, data):
        return data[0]
